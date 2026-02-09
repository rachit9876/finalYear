from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import get_paths


@dataclass
class Config:
    # Model identifiers or local paths (for offline use).
    text_model: str = "sentence-transformers/clip-ViT-B-32"
    image_model: str = "sentence-transformers/clip-ViT-B-32"

    # Optional dataset folder path (e.g. where your test files live)
    dataset_path: str = ""

    # OCR languages for tesseract (e.g. "eng", "hin", "eng+hin").
    # Requires matching system language data packages.
    ocr_languages: str = "eng"

    # Embedding vector size (used for index creation; must match model output).
    embedding_dim: int = 512

    # Maximum number of parallel workers for CPU-intensive tasks (ingestion, extraction, embedding).
    max_workers: int = 8


def load_config() -> Config:
    paths = get_paths()
    if not paths.config_path.exists():
        return Config()
    data = json.loads(paths.config_path.read_text(encoding="utf-8"))
    return Config(**data)


def save_config(config: Config) -> None:
    paths = get_paths()
    paths.config_path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")


def config_to_table(config: Config) -> list[tuple[str, Any]]:
    return [
        ("text_model", config.text_model),
        ("image_model", config.image_model),
        ("dataset_path", config.dataset_path),
        ("ocr_languages", config.ocr_languages),
        ("embedding_dim", config.embedding_dim),
    ]
