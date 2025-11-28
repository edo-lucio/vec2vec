import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import torch

from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.dist import get_rank
from utils.eval_utils import eval_loop_, text_to_embedding
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings

from itertools import combinations
import torch.nn.functional as F

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'{argv[1]}/config.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**cfg, **unknown_cfg})

    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        print("Note: bf16 is not available on this hardware!")

    random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())
    np.random.seed(cfg.seed + get_rank())
    torch.cuda.manual_seed(cfg.seed + get_rank())

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None
    )
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False
    dset = load_streaming_embeddings("laion")

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)}
    encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
    translator = load_n_translator(cfg, encoder_dims)

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)
    }
    unsup_dim = {
        cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb])
    }
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

    if cfg.style != 'identity':
        assert cfg.unsup_emb not in sup_encs
        assert cfg.unsup_emb in translator.in_adapters
        assert cfg.unsup_emb in translator.out_adapters

        cfg.num_params = sum(x.numel() for x in translator.parameters())
        print("Number of parameters:", cfg.num_params)

    num_workers = get_num_proc()
    evalset = MultiencoderTokenizedDataset(
        dataset=dset,
        encoders={ **unsup_enc, **sup_encs },
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )
    evalloader = DataLoader(
        evalset,
        batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    evalloader = accelerator.prepare(evalloader)

    if cfg.style != 'identity':
        assert cfg.unsup_emb not in sup_encs
        assert cfg.unsup_emb in translator.in_adapters
        assert cfg.unsup_emb in translator.out_adapters

        cfg.num_params = sum(x.numel() for x in translator.parameters())
        print("Number of parameters:", cfg.num_params)

    assert hasattr(cfg, 'num_points') or hasattr(cfg, 'unsup_points')

    if cfg.style != 'identity':
        assert hasattr(cfg, 'load_dir')
        print(f"Loading models from {argv[1]} on {accelerator.device}")
        translator.load_state_dict(torch.load(f'{argv[1]}/model.pt', map_location=accelerator.device), strict=False)

    print("Models Loaded")
    translator = accelerator.prepare(translator)
    sup_encoder = accelerator.prepare(sup_encs[cfg.sup_emb])
    unsup_encoder = accelerator.prepare(unsup_enc[cfg.unsup_emb])

    print("Models loaded on Accelerator")

    translator.eval()
    sup_encoder.eval()
    unsup_encoder.eval()

    with torch.no_grad():
        texts = [item["text"] for item in evalset.dataset]

        print(texts[:8])

        print("Inference")

        # Compute embeddings (stay on GPU)
        unsup_embs = text_to_embedding(
            texts, cfg.unsup_emb, unsup_enc[cfg.unsup_emb],
            cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device
        )
        sup_embs = text_to_embedding(
            texts, cfg.sup_emb, sup_encs[cfg.sup_emb],
            cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device
        )

        translated_unsup_embs, unsup_latents = translator.translate_embeddings(
            unsup_embs, in_name=cfg.unsup_emb, out_name=cfg.sup_emb
        )
        translated_sup_embs, sup_latents = translator.translate_embeddings(
            sup_embs, in_name=cfg.sup_emb, out_name=cfg.unsup_emb
        )

        print(translated_unsup_embs[0], unsup_latents[0])
        
        unsup_embs = unsup_embs.cpu()
        sup_embs = sup_embs.cpu()
        latents_unsup_embs = unsup_latents.cpu()
        latents_sup_embs = sup_latents.cpu()

    norm_unsup = F.normalize(unsup_embs, dim=1)
    norm_sup = F.normalize(sup_embs, dim=1)
    norm_trans_unsup = F.normalize(latents_unsup_embs, dim=1)
    norm_trans_sup = F.normalize(latents_sup_embs, dim=1)
    # norm_unsup = unsup_embs
    # norm_sup = sup_embs
    # norm_trans_unsup = latents_unsup_embs
    # norm_trans_sup = latents_sup_embs

    embs = {
        "unsup_pre_translation": norm_unsup,
        "sup_pre_translation": norm_sup,
        "unsup_latents": norm_trans_unsup,
        "sup_latents": norm_trans_sup,
    }

    N = len(texts)

    # Precompute indices
    idx_i, idx_j = torch.triu_indices(N, N, offset=1)

    output_path = f"similarities/{cfg.sup_emb}_{cfg.unsup_emb}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    chunk_size = 2000  # tune depending on GPU memory

    with open(output_path, "a") as f:
        buffer = []

        for start in range(0, len(idx_i), chunk_size):
            end = start + chunk_size
            batch_i = idx_i[start:end]
            batch_j = idx_j[start:end]

            # Compute chunked dot products for all embedding types
            results = {}
            for name, emb in embs.items():
                # Move only needed rows to GPU
                vi = emb[batch_i].to(accelerator.device)
                vj = emb[batch_j].to(accelerator.device)

                sims = (vi * vj).sum(dim=1).cpu()  # cosine since already normalized
                results[name] = sims

                del vi, vj, sims
                torch.cuda.empty_cache()

            # Write chunk to buffer
            for k in range(len(batch_i)):
                buffer.append(json.dumps({
                    "i": int(batch_i[k]),
                    "j": int(batch_j[k]),
                    "unsup_pre_translation": float(results["unsup_pre_translation"][k]),
                    "sup_pre_translation": float(results["sup_pre_translation"][k]),
                    "unsup_latents": float(results["unsup_latents"][k]),
                    "sup_latents": float(results["sup_latents"][k]),
                }))

            if len(buffer) > 5000:
                f.write("\n".join(buffer) + "\n")
                buffer = []

        if buffer:
            f.write("\n".join(buffer) + "\n")


if __name__ == "__main__":
    main()