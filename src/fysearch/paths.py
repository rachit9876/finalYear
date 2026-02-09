from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "input"

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "store"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def db_path(self) -> Path:
        return self.db_dir / "fysearch.sqlite3"

    @property
    def config_path(self) -> Path:
        return self.root / "fysearch.config.json"


def get_project_root() -> Path:
    # Assumes CLI is run from repo root (or a subdir). Walk upward until we find pyproject.
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def get_paths() -> ProjectPaths:
    return ProjectPaths(root=get_project_root())
