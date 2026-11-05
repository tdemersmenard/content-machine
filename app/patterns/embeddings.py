"""Embeddings locaux, stockés en float32 et normalisés : la similarité cosinus est un produit."""

from collections.abc import Callable
from functools import lru_cache

import numpy as np

from app.config import get_settings

Embedder = Callable[[list[str]], np.ndarray]


@lru_cache
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _load_model(get_settings().embedding_model)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)
