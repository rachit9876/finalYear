from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / norm


@dataclass
class TextEmbedder:
    model_name_or_path: str
    _model: Any = field(default=None, init=False, repr=False)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("sentence-transformers not installed; install extras: pip install -e .[embeddings]") from e
        self._model = SentenceTransformer(self.model_name_or_path)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        vec = np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts efficiently in batches. Returns array of shape (len(texts), dim)."""
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class ImageEmbedder:
    model_name_or_path: str
    _model: Any = field(default=None, init=False, repr=False)

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("sentence-transformers not installed; install extras: pip install -e .[embeddings]") from e
        self._model = SentenceTransformer(self.model_name_or_path)
        return self._model

    def embed(self, image_path: str) -> np.ndarray:
        from PIL import Image

        model = self._get_model()
        img = Image.open(image_path).convert("RGB")
        vec = np.asarray(model.encode([img], normalize_embeddings=True)[0], dtype=np.float32)
        return vec

    def embed_batch(self, image_paths: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed multiple images efficiently in batches. Returns array of shape (len(images), dim)."""
        from PIL import Image

        model = self._get_model()
        images = [Image.open(p).convert("RGB") for p in image_paths]
        vectors = model.encode(images, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def maybe_text_embedder(model_name_or_path: str) -> Optional[TextEmbedder]:
    if not model_name_or_path:
        return None
    return TextEmbedder(model_name_or_path=model_name_or_path)


def maybe_image_embedder(model_name_or_path: str) -> Optional[ImageEmbedder]:
    if not model_name_or_path:
        return None
    return ImageEmbedder(model_name_or_path=model_name_or_path)

