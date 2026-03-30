from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, List

import numpy as np


def _configure_torch_threads() -> None:
    """Configure PyTorch to use all available CPU cores for maximum throughput."""
    try:
        import torch
        num_cores = os.cpu_count() or 4
        # Intra-op parallelism (within operations like matrix multiply)
        torch.set_num_threads(num_cores)
        # Inter-op parallelism (across independent operations)
        try:
            torch.set_num_interop_threads(max(1, num_cores // 2))
        except RuntimeError:
            pass  # Can only be set once; ignore if already set
    except ImportError:
        pass


# Configure threading on module load — this must happen before any model inference.
_configure_torch_threads()


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
        try:
            import torch
            with torch.inference_mode():
                vec = np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        except ImportError:
            vec = np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed multiple texts efficiently in batches. Returns array of shape (len(texts), dim)."""
        model = self._get_model()
        try:
            import torch
            with torch.inference_mode():
                vectors = model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
        except ImportError:
            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
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
        try:
            import torch
            with torch.inference_mode():
                vec = np.asarray(model.encode([img], normalize_embeddings=True)[0], dtype=np.float32)
        except ImportError:
            vec = np.asarray(model.encode([img], normalize_embeddings=True)[0], dtype=np.float32)
        return vec

    def embed_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple images efficiently in batches. Returns array of shape (len(images), dim)."""
        from PIL import Image

        model = self._get_model()
        images = [Image.open(p).convert("RGB") for p in image_paths]
        try:
            import torch
            with torch.inference_mode():
                vectors = model.encode(
                    images,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
        except ImportError:
            vectors = model.encode(
                images,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)


def maybe_text_embedder(model_name_or_path: str) -> Optional[TextEmbedder]:
    if not model_name_or_path:
        return None
    return TextEmbedder(model_name_or_path=model_name_or_path)


def maybe_image_embedder(model_name_or_path: str) -> Optional[ImageEmbedder]:
    if not model_name_or_path:
        return None
    return ImageEmbedder(model_name_or_path=model_name_or_path)
