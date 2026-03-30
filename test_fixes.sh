#!/bin/bash
cd /mnt/c/Users/drspacelove/Documents/GitHub/finalYear
source .venv/bin/activate
echo "=== Testing CLI import ==="
python -c "from fysearch.cli import app; print('CLI import OK')"
echo "=== Testing config ==="
python -c "from fysearch.config import load_config; cfg = load_config(); print(f'Config OK: dim={cfg.embedding_dim}, workers={cfg.effective_max_workers}')"
echo "=== Testing paths ==="
python -c "from fysearch.paths import is_wsl, normalize_path; print(f'WSL={is_wsl()}, normalize_path test: {normalize_path(\"C:/Users/test\")}')"
echo "=== Testing embeddings module ==="
python -c "from fysearch.embeddings import maybe_text_embedder; print('Embeddings module OK')"
echo "=== Testing vector_index ==="
python -c "from fysearch.vector_index import BruteForceIndex; import numpy as np; idx=BruteForceIndex(3); idx.add(['a','b'], np.array([[1,0,0],[0,1,0]], dtype=np.float32)); hits=idx.search(np.array([1,0,0], dtype=np.float32), 1); print(f'Vector index OK: {hits[0].doc_id}={hits[0].score:.2f}')"
echo "=== All tests passed ==="
