"""
Local embedding service using Qwen3-Embedding-0.6B via mlx-embeddings.

Uses Apple MLX for efficient inference on Apple Silicon.
Lazily loads the model on first use. All subsequent calls reuse the
loaded model, so the load cost is paid only once.
"""

from __future__ import annotations

import threading

import mlx.core as mx
from mlx_embeddings.utils import load as mlx_load

MODEL_ID = "mlx-community/Qwen3-Embedding-0.6B-mxfp8"
EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 4096

_model = None
_tokenizer = None
_model_lock = threading.Lock()


def _get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    with _model_lock:
        if _model is None:
            _model, _tokenizer = mlx_load(MODEL_ID)
        return _model, _tokenizer


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts and return normalised 1024-d vectors as Python lists."""
    model, tokenizer = _get_model()
    tok = tokenizer._tokenizer
    encoded = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )
    input_ids = mx.array(encoded["input_ids"])
    attention_mask = mx.array(encoded["attention_mask"])
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    mx.eval(output.text_embeds)
    return output.text_embeds.tolist()


def embed(texts: list[str]) -> list[list[float]]:
    """Embed one or more texts, returning normalised 1024-d vectors."""
    if not texts:
        return []
    return _embed(texts)


def embed_query(query: str) -> list[float]:
    """Embed a single search query."""
    return _embed([query])[0]


def embed_documents(docs: list[str]) -> list[list[float]]:
    """Embed documents."""
    return embed(docs)
