import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import torch

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.dist import get_rank
from utils.eval_utils import eval_loop_, text_to_embedding
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings

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

    # # === 🔹 ADD THIS BLOCK BELOW 🔹 ===
    # from transformers import CLIPTokenizerFast
    # import re
    # print("Filtering dataset for CLIP-compatible text...")
    # # Initialize CLIP tokenizer
    # clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    # # Length-based filter
    # def filter_length(batch, max_tokens=77, min_tokens=3):
    #     token_counts = [len(clip_tokenizer(t, truncation=False)["input_ids"]) for t in batch["text"]]
    #     mask = [(min_tokens <= c <= max_tokens) for c in token_counts]
    #     return {k: [v[i] for i in range(len(v)) if mask[i]] for k, v in batch.items()}

    # dset = dset.map(filter_length, batched=True, batch_size=256)
    # print(f"After token length filter: {len(dset)} samples")

    # # Clean text filter (remove URLs, code, metadata, etc.)
    # def clean_text(text):
    #     text = text.strip()
    #     if len(text.split()) < 3:
    #         return False
    #     if re.search(r"https?://|www\.|@|#|\{|\}|\[|\]|<|>|=", text):
    #         return False
    #     if re.match(r"^[0-9\W_]+$", text):
    #         return False
    #     if any(sym in text for sym in ["<table>", "<ref>", "Category:", "ISBN", "doi:"]):
    #         return False
    #     return True

    # dset = dset.filter(lambda x: clean_text(x["text"]))
    # print(f"After cleaning filter: {len(dset)} samples")

    # from sentence_transformers import SentenceTransformer, util
    # caption_model = SentenceTransformer("clip-ViT-B-32")
    # caption_ref = caption_model.encode(
    #     ["a person doing something", "an object in a scene", "a close-up of something"],
    #     normalize_embeddings=True
    # )
    
    # def is_caption_like(text):
    #     emb = caption_model.encode(text, normalize_embeddings=True)
    #     sim = util.cos_sim(emb, caption_ref).max().item()
    #     return sim > 0.25  # tune threshold if needed
    
    # dset = dset.filter(lambda x: is_caption_like(x["text"]))
    # print(f"After caption-like filter: {len(dset)} samples")

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

    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    dset = dset_dict["train"]
    valset = dset_dict["test"]

    assert hasattr(cfg, 'num_points') or hasattr(cfg, 'unsup_points')
    dset = dset.shuffle(seed=cfg.train_dataset_seed)
    if hasattr(cfg, 'num_points'):
        assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
        supset = dset.select(range(cfg.num_points))
        unsupset = dset.select(range(cfg.num_points, cfg.num_points + cfg.val_size))
    elif hasattr(cfg, 'unsup_points'):
        unsupset = dset.select(range(min(cfg.unsup_points, cfg.val_size)))
        supset = dset.select(range(min(cfg.unsup_points, len(dset)), len(dset) - len(unsupset)))

    num_workers = get_num_proc()
    evalset = MultiencoderTokenizedDataset(
        dataset=supset if hasattr(cfg, 'flip') and cfg.flip else unsupset,
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
        print(f"Loading models from {argv[1]}...")
        translator.load_state_dict(torch.load(f'{argv[1]}/model.pt', map_location='cpu'), strict=False)

    translator = accelerator.prepare(translator)
    sup_encoder = accelerator.prepare(sup_encs[cfg.sup_emb])
    unsup_encoder = accelerator.prepare(unsup_enc[cfg.unsup_emb])

    # inverters = get_inverters(["gtr"], accelerator.device)
    inverters = None

    from itertools import combinations
    import torch.nn.functional as F

    translator.eval()
    sup_encoder.eval()
    unsup_encoder.eval()

    cosine_similarities = {}  # dictionary to store cosine similarities

    with torch.no_grad():
        # Collect all texts from the evalset
        texts = [item["text"] for item in evalset.dataset] if hasattr(evalset, "dataset") else evalset["text"]

        # Get embeddings for all items
        unsup_embs = text_to_embedding(
            texts, cfg.unsup_emb, unsup_enc[cfg.unsup_emb],
            cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device
        )
        sup_embs = text_to_embedding(
            texts, cfg.sup_emb, sup_encs[cfg.sup_emb],
            cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device
        )

        # Translate all unsupervised embeddings
        translated_unsup_embs = translator.translate_embeddings(
            unsup_embs, in_name=cfg.unsup_emb, out_name=cfg.sup_emb
        )

        translated_sup_embs = translator.translate_embeddings(
            sup_embs, in_name=cfg.unsup_emb, out_name=cfg.sup_emb
        )

        unsup_embs = unsup_embs.cpu()
        sup_embs = sup_embs.cpu()
        translated_unsup_embs = translated_unsup_embs.cpu()
        translated_sup_embs = translated_sup_embs.cpu()

        output_path = f"similarities/{cfg.sup_emb}_{cfg.unsup_emb}.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a") as f:
        for i, j in combinations(range(len(texts)), 2):
            result = {
                "pair_key": f"{texts[i]}_{texts[j]}",
                "unsup_pre_translation": F.cosine_similarity(unsup_embs[i].unsqueeze(0), unsup_embs[j].unsqueeze(0)).item(),
                "sup_pre_translation": F.cosine_similarity(sup_embs[i].unsqueeze(0), sup_embs[j].unsqueeze(0)).item(),
                "unsup_translated": F.cosine_similarity(translated_unsup_embs[i].unsqueeze(0), translated_unsup_embs[j].unsqueeze(0)).item(),
                "sup_translated": F.cosine_similarity(translated_sup_embs[i].unsqueeze(0), translated_sup_embs[j].unsqueeze(0)).item(),
            }
            f.write(json.dumps(result) + "\n")
            f.flush()  # ensure data is written immediately
            os.fsync(f.fileno())  # optional: forces write to disk

if __name__ == "__main__":
    main()