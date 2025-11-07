from typing import Optional

import torch
from sentence_transformers import SentenceTransformer, models
from utils.st_wrapper import AudioSentenceTransformer

MODEL_PATH = '/private/home/jxm/supervised_translation/model_weights/'

HF_FLAGS = {
    'gtr': 'sentence-transformers/gtr-t5-base',
    'gte': 'thenlper/gte-base',
    'gist': 'avsolatorio/GIST-Embedding-v0',
    'stella': 'infgrad/stella-base-en-v2',
    'sentence-t5': 'sentence-transformers/sentence-t5-base',
    'e5': 'intfloat/e5-base-v2',
    'ember': 'llmrails/ember-v1',
    'snowflake': 'Snowflake/snowflake-arctic-embed-m-long',
    'sbert': 'sentence-transformers/all-MiniLM-L12-v2',
    'clip': 'sentence-transformers/clip-ViT-B-32',
    'jina': 'jinaai/jina-embeddings-v2-base-en',
    'bert-nli': 'sentence-transformers/bert-base-nli-mean-tokens',
    'dpr': 'sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base',
    'granite': 'ibm-granite/granite-embedding-278m-multilingual',
    'modernbert': 'answerdotai/ModernBERT-base',
    'modernbert-large': 'answerdotai/ModernBERT-large',
    'nomicbert': 'nomic-ai/nomic-bert-2048',
    'qwen': 'Qwen/Qwen3-Embedding-0.6B',
    'clap': 'laion/clap-htsat-unfused',
}

def load_encoder(model_flag: str, device: str = 'cpu', mixed_precision: Optional[str] = None):
    model_id = HF_FLAGS.get(model_flag, model_flag)

    # Setup dtype for mixed precision
    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'no': torch.float32}
    model_kwargs = {'torch_dtype': dtype_map.get(mixed_precision, torch.float32)}

    # Special handling for GPT-2 variants
    if model_flag.startswith("gpt2"):
        transformer = models.Transformer("sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256)
        normalize = models.Normalize()
        pooling_mode = "mean" if model_flag == "gpt2_mean" else "lasttoken"
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling_mode)
        encoder = SentenceTransformer(modules=[transformer, pooling, normalize])

    # Special handling for CLAP (audio-text)
    elif model_flag.startswith("clap"):
        encoder = AudioSentenceTransformer(model_id)

    # Special handling for CLIP text embeddings
    elif model_flag.startswith("clip"):
        # Load SentenceTransformer CLIP model (text encoder works automatically for text)
        encoder = SentenceTransformer(model_id, device=device, trust_remote_code=True, model_kwargs=model_kwargs)

    # Default case: load any Hugging Face/SentenceTransformer model
    else:
        encoder = SentenceTransformer(model_id, device=device, trust_remote_code=True, model_kwargs=model_kwargs)

    return encoder.eval()


def get_sentence_embedding_dimension(encoder):
    dim = encoder.get_sentence_embedding_dimension()
    if dim is not None:
        return dim

    # special handling for CLIP models
    dim = encoder[0].model.text_model.config.hidden_size

    return dim