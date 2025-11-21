import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
from numpy import dot
from numpy.linalg import norm

import torch

from utils.dist import get_rank
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.eval_utils import text_to_embedding

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'{argv[1]}/config.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**cfg, **unknown_cfg})

    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        print("Note: bf16 is not available on this hardware!")

    # set seeds
    random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())
    np.random.seed(cfg.seed + get_rank())
    torch.cuda.manual_seed(cfg.seed + get_rank())

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None
    )
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False

    sup_encs = { cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None) }
    encoder_dims = { cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb]) }
    translator = load_n_translator(cfg, encoder_dims)

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = { cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None) }
    unsup_dim = { cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb]) }
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

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

    translator.eval()
    sup_encoder.eval()
    unsup_encoder.eval()

    with torch.no_grad():
        unsup_emb_1 = text_to_embedding(["hissing"], cfg.unsup_emb, unsup_enc[cfg.unsup_emb], cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device)
        unsup_emb_2 = text_to_embedding(["air leak"], cfg.unsup_emb, unsup_enc[cfg.unsup_emb], cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device)
        unsup_emb_3 = text_to_embedding(["fortune"], cfg.unsup_emb, unsup_enc[cfg.unsup_emb], cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device)

        sup_emb_1 = text_to_embedding(["hissing"], cfg.sup_emb, sup_encs[cfg.sup_emb], cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device).squeeze().cpu()
        sup_emb_2 = text_to_embedding(["air leak"], cfg.sup_emb, sup_encs[cfg.sup_emb], cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device).squeeze().cpu()
        sup_emb_3 = text_to_embedding(["fortune"], cfg.sup_emb, sup_encs[cfg.sup_emb], cfg.normalize_embeddings, cfg.max_seq_length, accelerator.device).squeeze().cpu()

        # unsup_emb_1 = unsup_encoder(["thunder"], return_tensors="pt")
        # unsup_emb_2 = unsup_encoder(["bass"], return_tensors="pt")        
        translated_emb_1 = translator.translate_embeddings(unsup_emb_1, in_name=cfg.unsup_emb, out_name=cfg.sup_emb).squeeze().cpu()
        translated_emb_2 = translator.translate_embeddings(unsup_emb_2, in_name=cfg.unsup_emb, out_name=cfg.sup_emb).squeeze().cpu()
        translated_emb_3 = translator.translate_embeddings(unsup_emb_3, in_name=cfg.unsup_emb, out_name=cfg.sup_emb).squeeze().cpu()

        unsup_emb_1 = unsup_emb_1.squeeze().cpu()
        unsup_emb_2 = unsup_emb_2.squeeze().cpu()
        unsup_emb_3 = unsup_emb_3.squeeze().cpu()

        pre_control_cos_sim = dot(sup_emb_1, sup_emb_2)/(norm(sup_emb_1)*norm(sup_emb_2))
        print(f"Supervised Pre-Translation Cosine Similarity {pre_control_cos_sim} \n")

        pre_cos_sim = dot(unsup_emb_1, unsup_emb_2)/(norm(unsup_emb_1)*norm(unsup_emb_2))
        print(f"Unsupervsied Pre-Translation Cosine Similarity {pre_cos_sim} \n")

        pre_control_cos_sim = dot(sup_emb_1, sup_emb_3)/(norm(sup_emb_1)*norm(sup_emb_3))
        print(f"Control Supervised Pre-Translation Cosine Similarity {pre_control_cos_sim} \n")

        pre_control_cos_sim = dot(unsup_emb_1, unsup_emb_3)/(norm(unsup_emb_1)*norm(unsup_emb_3))
        print(f"Control Unsupervised Pre-Translation Cosine Similarity {pre_control_cos_sim} \n")

        translated_control_cos_sim = dot(translated_emb_1, translated_emb_3)/(norm(translated_emb_1)*norm(translated_emb_3))
        print(f"Control Unsupervised Translated Cosine Similarity {translated_control_cos_sim} \n")

        translated_cos_sim = dot(translated_emb_1, translated_emb_2)/(norm(translated_emb_1)*norm(translated_emb_2))
        print(f"UnsuperviseTranslated Cosine Similarity {translated_cos_sim} \n")


if __name__ == "__main__":
    main()