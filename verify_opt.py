
import os
import shutil
from pathlib import Path
import time
from unittest import mock
from fysearch.cli import app
from fysearch.paths import ProjectPaths
from typer.testing import CliRunner

runner = CliRunner()

TEST_ROOT = Path("./test_data_opt").resolve()

def setup_test_data():
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True)
    
    # Create valid project structure
    (TEST_ROOT / "pyproject.toml").touch()
    
    # Create input data
    input_dir = TEST_ROOT / "input"
    input_dir.mkdir()
    
    # Create some dummy text files
    for i in range(50):
        (input_dir / f"doc_{i}.txt").write_text(f"This is document number {i}. It has some content related to optimization.", encoding="utf-8")

def mock_get_paths():
    return ProjectPaths(root=TEST_ROOT)

def test_workflow():
    setup_test_data()
    
    # Patch get_paths in fysearch.paths so all modules see the mock
    with mock.patch("fysearch.paths.get_paths", side_effect=mock_get_paths):
        print(f"--- Testing in {TEST_ROOT} ---")
        
        print("--- Init ---")
        result = runner.invoke(app, ["init"])
        print(result.stdout)
        if result.exit_code != 0:
            print(result.stderr)
            return
            
        print("--- Ingest ---")
        # Ingest the input directory we created
        start = time.time()
        result = runner.invoke(app, ["ingest", str(TEST_ROOT / "input")])
        print(f"Ingest took: {time.time() - start:.2f}s")
        print(result.stdout)
        if result.exit_code != 0:
            print(result.stderr)
            return

        print("--- Extract ---")
        start = time.time()
        result = runner.invoke(app, ["extract"])
        print(f"Extract took: {time.time() - start:.2f}s")
        print(result.stdout)
        if result.exit_code != 0:
            print(result.stderr)
            return

        print("--- Build Index ---")
        start = time.time()
        result = runner.invoke(app, ["build-index", "--modality", "text"])
        print(f"Build Index took: {time.time() - start:.2f}s")
        print(result.stdout)
        if result.exit_code != 0:
            print(result.stderr)
            return

        print("--- Search ---")
        result = runner.invoke(app, ["search-text", "optimization", "--top-k", "5"])
        print(result.stdout)
        if result.exit_code != 0:
            print(result.stderr)
            
    # Cleanup
    # shutil.rmtree(TEST_ROOT)

if __name__ == "__main__":
    test_workflow()
