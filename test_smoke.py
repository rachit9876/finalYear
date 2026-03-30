#!/usr/bin/env python3
"""Quick smoke test for all fysearch modules after bug fixes."""
import sys
import os

# Ensure we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=== Testing module imports ===")
try:
    from fysearch.paths import ProjectPaths, get_project_root, is_wsl, normalize_path
    print(f"  paths: OK (WSL={is_wsl()})")
    print(f"  normalize_path('C:/Users/test') = {normalize_path('C:/Users/test')}")
    print(f"  normalize_path('/mnt/c/Users/test') = {normalize_path('/mnt/c/Users/test')}")
except Exception as e:
    print(f"  paths: FAIL - {e}")
    sys.exit(1)

try:
    from fysearch.config import Config, load_config, config_to_table
    cfg = Config()
    print(f"  config: OK (dim={cfg.embedding_dim}, workers=effective:{cfg.effective_max_workers})")
    table = config_to_table(cfg)
    assert any('max_workers' in str(k) for k, v in table), "max_workers missing from config_to_table!"
    print(f"  config_to_table: OK ({len(table)} entries)")
except Exception as e:
    print(f"  config: FAIL - {e}")
    sys.exit(1)

try:
    from fysearch.db import connect, init_db
    print("  db: OK")
except Exception as e:
    print(f"  db: FAIL - {e}")

try:
    from fysearch.embeddings import maybe_text_embedder, maybe_image_embedder, TextEmbedder, ImageEmbedder
    print("  embeddings: OK")
except Exception as e:
    print(f"  embeddings: FAIL - {e}")

try:
    from fysearch.vector_index import BruteForceIndex, SearchHit
    import numpy as np
    idx = BruteForceIndex(3)
    idx.add(['a', 'b'], np.array([[1,0,0],[0,1,0]], dtype=np.float32))
    hits = idx.search(np.array([1,0,0], dtype=np.float32), 1)
    assert len(hits) == 1
    assert hits[0].doc_id == 'a'
    assert hits[0].score > 0.99
    print(f"  vector_index: OK (search hit: {hits[0].doc_id}={hits[0].score:.2f})")
except Exception as e:
    print(f"  vector_index: FAIL - {e}")
    sys.exit(1)

try:
    from fysearch.ingest import _detect_media_type, _hash_file
    from pathlib import Path
    assert _detect_media_type(Path("test.jpg")) == "image"
    assert _detect_media_type(Path("test.pdf")) == "pdf"
    assert _detect_media_type(Path("test.txt")) == "text"
    print("  ingest: OK")
except Exception as e:
    print(f"  ingest: FAIL - {e}")

try:
    from fysearch.extract import extract_pdf_text, ocr_image
    print("  extract: OK")
except Exception as e:
    print(f"  extract: FAIL - {e}")

try:
    from fysearch.cli import app, _extract_worker
    print("  cli: OK (app and _extract_worker imported)")
except Exception as e:
    print(f"  cli: FAIL - {e}")
    sys.exit(1)

try:
    from fysearch.webapp import create_app, text_query, image_query, auto_query
    print("  webapp: OK")
except Exception as e:
    print(f"  webapp: FAIL - {e}")

print("\n=== All module import tests passed! ===")
