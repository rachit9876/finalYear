from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import shutil

import numpy as np

from .config import load_config
from .db import add_search_history, clear_search_history, connect, init_db, list_documents, list_search_history
from .embeddings import maybe_image_embedder, maybe_text_embedder
from .extract import extract_text_for_doc
from .ingest import ingest_path
from .paths import get_paths, normalize_path
from .vector_index import BruteForceIndex, FaissIndex


@dataclass(frozen=True)
class ResultRow:
    score: float
    doc_id: str
    stored_path: str
    original_path: str
    media_type: str
    method: str
    snippet: str
    page: str


def _page_from_original_path(original_path: str) -> str:
    marker = "#page="
    if marker not in original_path:
        return ""
    try:
        return original_path.split(marker, 1)[1].strip()
    except Exception:
        return ""


def _make_snippet(text: str, query: str, max_len: int = 220) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    q = (query or "").strip().lower()
    if not q:
        return (text[:max_len] + ("…" if len(text) > max_len else "")).strip()

    # Try to anchor around the first matching token.
    tokens = [t for t in q.replace("\n", " ").split() if len(t) >= 3]
    hay = text.lower()
    idx = -1
    for tok in tokens[:8]:
        idx = hay.find(tok)
        if idx != -1:
            break
    if idx == -1:
        return (text[:max_len] + ("…" if len(text) > max_len else "")).strip()

    start = max(0, idx - max_len // 2)
    end = min(len(text), start + max_len)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _get_index(dim: int, prefer_faiss: bool):
    if prefer_faiss:
        try:
            return FaissIndex(dim)
        except Exception:
            return BruteForceIndex(dim)
    return BruteForceIndex(dim)


def _load_npz(modality: str) -> tuple[list[str], np.ndarray]:
    paths = get_paths()
    name = "text_index.npz" if modality == "text" else "image_index.npz"
    npz = paths.index_dir / name
    if not npz.exists():
        raise FileNotFoundError(f"Missing index: {npz}")
    data = np.load(npz, allow_pickle=True)
    doc_ids = [str(x) for x in data["doc_ids"]]
    vectors = data["vectors"].astype(np.float32)
    return doc_ids, vectors


def _build_text_index(prefer_faiss: bool) -> Path:
    cfg = load_config()
    embedder = maybe_text_embedder(cfg.text_model)
    if embedder is None:
        raise RuntimeError("Config.text_model is empty")

    conn = connect()
    init_db(conn)
    rows = conn.execute(
        """
        SELECT d.doc_id, t.text
        FROM documents d
        JOIN extracted_text t ON t.doc_id = d.doc_id
        WHERE t.text IS NOT NULL
          AND length(trim(t.text)) > 0
          AND (
            -- Include per-page docs (they have #page= in original_path)
            d.original_path LIKE '%#page=%'
            -- Include non-PDF docs
            OR d.media_type NOT IN ('pdf', 'image')
            -- Include PDFs without per-page docs
            OR (
                d.media_type = 'pdf'
                AND NOT EXISTS (
                    SELECT 1
                    FROM documents p
                    WHERE p.original_path LIKE d.original_path || '#page=%'
                )
            )
          )
        """
    ).fetchall()
    conn.close()

    # Prepare data for batch embedding
    doc_ids: list[str] = []
    texts: list[str] = []
    for r in rows:
        text = (r["text"] or "").strip()
        if text:
            doc_ids.append(r["doc_id"])
            texts.append(text)

    if not texts:
        raise RuntimeError("No extracted text available to index. Enable OCR or ingest PDFs with selectable text.")

    # Batch embedding - much faster than one-by-one or threading
    mat = embedder.embed_batch(texts, batch_size=32)

    paths = get_paths()
    out = paths.index_dir / "text_index.npz"
    np.savez_compressed(out, doc_ids=np.asarray(doc_ids), vectors=mat)
    return out


def _build_image_index(prefer_faiss: bool) -> Path:
    cfg = load_config()
    embedder = maybe_image_embedder(cfg.image_model)
    if embedder is None:
        raise RuntimeError("Config.image_model is empty")

    conn = connect()
    init_db(conn)
    rows = conn.execute(
        """
        SELECT doc_id, stored_path
        FROM documents
        WHERE media_type = 'image'
        """
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No images found to index.")

    doc_ids = [r["doc_id"] for r in rows]
    image_paths = [r["stored_path"] for r in rows]

    # Batch embedding - much faster than one-by-one or threading
    mat = embedder.embed_batch(image_paths, batch_size=16)

    paths = get_paths()
    out = paths.index_dir / "image_index.npz"
    np.savez_compressed(out, doc_ids=np.asarray(doc_ids), vectors=mat)
    return out


def text_query(query: str, target_modality: str, top_k: int, prefer_faiss: bool) -> list[ResultRow]:
    cfg = load_config()
    text_embedder = maybe_text_embedder(cfg.text_model)
    if text_embedder is None:
        raise RuntimeError("Config.text_model is empty")

    doc_ids, vectors = _load_npz(target_modality)
    dim = vectors.shape[1]

    idx = _get_index(dim=dim, prefer_faiss=prefer_faiss)
    idx.add(doc_ids, vectors)

    q = text_embedder.embed(query).astype(np.float32)
    # Fetch more results for re-ranking with keyword boost
    hits = idx.search(q, min(top_k * 3, len(doc_ids)))

    conn = connect()
    init_db(conn)

    # Extract query keywords for boosting (lowercase, strip punctuation)
    query_keywords = [w.lower().strip("?!.,;:\"'") for w in query.split() if len(w) > 2]

    results: list[tuple[float, ResultRow]] = []
    for hit in hits:
        row = conn.execute(
            """
            SELECT d.stored_path, d.original_path, d.media_type, t.method, t.text
            FROM documents d
            LEFT JOIN extracted_text t ON t.doc_id = d.doc_id
            WHERE d.doc_id = ?
            """,
            (hit.doc_id,),
        ).fetchone()
        if not row:
            continue

        original_path = str(row["original_path"])
        text = str(row["text"] or "")
        text_lower = text.lower()
        
        # Calculate keyword boost: +0.5 for each query keyword found in text
        keyword_boost = sum(0.5 for kw in query_keywords if kw in text_lower)
        boosted_score = hit.score + keyword_boost
        
        result = ResultRow(
            score=boosted_score,
            doc_id=hit.doc_id,
            stored_path=str(row["stored_path"]),
            original_path=original_path,
            media_type=str(row["media_type"]),
            method=str(row["method"] or ""),
            snippet=_make_snippet(text, query=query),
            page=_page_from_original_path(original_path),
        )
        results.append((boosted_score, result))

    conn.close()
    
    # Sort by boosted score (highest first) and return top_k
    results.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in results[:top_k]]


def auto_query(query: str, top_k: int, prefer_faiss: bool) -> tuple[list[ResultRow], list[ResultRow]]:
    """Smart auto search: returns separate text and image results for side-by-side display."""
    text_results: list[ResultRow] = []
    image_results: list[ResultRow] = []
    
    # Try text search
    try:
        text_results = text_query(query=query, target_modality="text", top_k=top_k, prefer_faiss=prefer_faiss)
    except (FileNotFoundError, RuntimeError):
        pass  # Text index might not exist
    
    # Try image search
    try:
        image_results = text_query(query=query, target_modality="image", top_k=top_k, prefer_faiss=prefer_faiss)
    except (FileNotFoundError, RuntimeError):
        pass  # Image index might not exist
    
    return (text_results, image_results)


def image_query(image_path: Path, top_k: int, prefer_faiss: bool) -> list[ResultRow]:
    cfg = load_config()
    image_embedder = maybe_image_embedder(cfg.image_model)
    if image_embedder is None:
        raise RuntimeError("Config.image_model is empty")

    doc_ids, vectors = _load_npz("image")
    dim = vectors.shape[1]

    idx = _get_index(dim=dim, prefer_faiss=prefer_faiss)
    idx.add(doc_ids, vectors)

    q = image_embedder.embed(str(image_path)).astype(np.float32)
    hits = idx.search(q, top_k)

    conn = connect()
    init_db(conn)

    out: list[ResultRow] = []
    for hit in hits:
        row = conn.execute(
            "SELECT stored_path, original_path, media_type FROM documents WHERE doc_id = ?",
            (hit.doc_id,),
        ).fetchone()
        if not row:
            continue
        original_path = str(row["original_path"])
        out.append(
            ResultRow(
                score=hit.score,
                doc_id=hit.doc_id,
                stored_path=str(row["stored_path"]),
                original_path=original_path,
                media_type=str(row["media_type"]),
                method="",
                snippet="",
                page=_page_from_original_path(original_path),
            )
        )

    conn.close()
    return out


def create_app():
    try:
        from flask import Flask, abort, flash, get_flashed_messages, redirect, render_template, request, send_file, url_for
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Flask not installed. Run: pip install -e '.[web]'") from e

    app = Flask(__name__)
    app.secret_key = 'fysearch-dev-key-change-in-production'

    def _last_uploaded_filename_from_history(history: list[dict[str, str]]) -> Optional[str]:
        for h in history:
            if (h.get("kind") or "").lower() != "image":
                continue
            p = (h.get("uploaded_path") or "").strip()
            if not p:
                continue
            name = Path(p).name
            if not name or Path(name).name != name:
                continue
            return name
        return None

    @app.get("/upload/<name>")
    def serve_upload(name: str):
        # Serve uploaded query images from data/uploads only.
        if not name or Path(name).name != name:
            abort(404)
        paths = get_paths()
        uploads_root = (paths.data_dir / "uploads").resolve()
        fp = (uploads_root / name).resolve()
        if uploads_root not in fp.parents or not fp.exists() or not fp.is_file():
            abort(404)
        return send_file(fp)

    def _reset_all_data() -> None:
        """Wipe local data so the project starts fresh."""
        paths = get_paths()

        # Stop serving stale paths by clearing config dataset_path.
        cfg = load_config()
        if getattr(cfg, "dataset_path", ""):
            cfg.dataset_path = ""
            from .config import save_config

            save_config(cfg)

        # Remove derived data
        for folder in [paths.store_dir, paths.index_dir, paths.data_dir / "uploads"]:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)

        # Remove DB file
        if paths.db_path.exists():
            try:
                paths.db_path.unlink()
            except Exception:
                pass

        # Recreate base folders
        paths.store_dir.mkdir(parents=True, exist_ok=True)
        paths.index_dir.mkdir(parents=True, exist_ok=True)
        (paths.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        paths.db_dir.mkdir(parents=True, exist_ok=True)

        # Re-init empty DB
        conn = connect()
        init_db(conn)
        conn.close()

    def _index_status() -> dict[str, bool]:
        paths = get_paths()
        return {
            "has_text_index": (paths.index_dir / "text_index.npz").exists(),
            "has_image_index": (paths.index_dir / "image_index.npz").exists(),
        }

    def _get_history() -> list[dict[str, str]]:
        conn = connect()
        init_db(conn)
        rows = list_search_history(conn, limit=25)
        conn.close()
        out: list[dict[str, str]] = []
        for r in rows:
            out.append(
                {
                    "created_at": str(r["created_at"]),
                    "kind": str(r["kind"]),
                    "query_text": str(r["query_text"] or ""),
                    "target_modality": str(r["target_modality"] or ""),
                    "top_k": str(r["top_k"] or ""),
                    "uploaded_path": str(r["uploaded_path"] or ""),
                }
            )
        return out

    @app.get("/")
    def index():
        # Clear history on every page load for a fresh start
        conn = connect()
        init_db(conn)
        clear_search_history(conn)
        conn.close()
        
        cfg = load_config()
        history = _get_history()
        # Check for flash messages from POST redirects
        flashes = get_flashed_messages(with_categories=True)
        error = None
        message = None
        for category, msg in flashes:
            if category == 'error':
                error = msg
            else:
                message = msg
        return render_template(
            "index.html",
            text_results=None,
            image_results=None,
            error=error,
            message=message,
            query="",
            top_k=1,
            modality="image",
            dataset_path=cfg.dataset_path,
            history=history,
            last_uploaded_filename=_last_uploaded_filename_from_history(history),
            **_index_status(),
        )

    @app.post("/history/clear")
    def clear_history():
        conn = connect()
        init_db(conn)
        uploaded_paths = clear_search_history(conn)
        conn.close()

        # Best-effort delete uploaded query images (only within data/uploads)
        paths = get_paths()
        uploads_root = (paths.data_dir / "uploads").resolve()
        deleted = 0
        for p in uploaded_paths:
            try:
                fp = Path(p).resolve()
                if uploads_root in fp.parents and fp.exists() and fp.is_file():
                    fp.unlink()
                    deleted += 1
            except Exception:
                continue

        flash(f"Cleared history. Deleted {deleted} uploaded query images.", "success")
        return redirect(url_for("index"))

    @app.post("/reset")
    def reset():
        confirm = (request.form.get("confirm") or "").strip().lower() == "on"
        if not confirm:
            flash("Reset not confirmed. Tick the checkbox and try again.", "error")
            return redirect(url_for("index"))

        _reset_all_data()
        flash("Reset complete. Local store/index/DB/uploads cleared.", "success")
        return redirect(url_for("index"))

    @app.post("/build-index")
    def build_index_route():
        modality = (request.form.get("modality") or "image").strip().lower()
        prefer_faiss = True

        if modality not in {"text", "image"}:
            flash("Invalid modality. Choose text or image.", "error")
            return redirect(url_for("index"))

        try:
            if modality == "image":
                _build_image_index(prefer_faiss=prefer_faiss)
                flash("Image index built successfully!", "success")
            else:
                _build_text_index(prefer_faiss=prefer_faiss)
                flash("Text index built successfully!", "success")
        except Exception as e:
            flash(f"Failed to build {modality} index: {str(e)}", "error")

        return redirect(url_for("index"))

    @app.post("/dataset")
    def set_dataset():
        cfg = load_config()
        dataset_path = (request.form.get("dataset_path") or "").strip()
        run_pipeline = (request.form.get("run_pipeline") or "").strip().lower() == "on"
        prefer_faiss = True

        if not dataset_path:
            flash("Dataset folder path is empty", "error")
            return redirect(url_for("index"))

        try:
            # Automatically convert Windows paths to WSL paths when running under WSL
            dataset_path = normalize_path(dataset_path)
            
            p = Path(dataset_path).expanduser()
            
            if not p.exists():
                flash(f"Folder does not exist: {dataset_path}", "error")
                return redirect(url_for("index"))
            if not p.is_dir():
                flash(f"Path is not a directory: {dataset_path}", "error")
                return redirect(url_for("index"))
            
            # Resolve after validation
            p = p.resolve()
        except Exception as e:
            flash(f"Invalid path: {dataset_path} - {str(e)}", "error")
            return redirect(url_for("index"))

        # Save to config
        cfg.dataset_path = str(p)
        from .config import save_config

        save_config(cfg)

        if run_pipeline:
            try:
                conn = connect()
                init_db(conn)
                ingest_results = ingest_path(conn, p)
                new_files = sum(1 for r in ingest_results if r.is_new)
                existing_files = len(ingest_results) - new_files
                # Extract for all docs (new ones + existing)
                extracted = 0
                for row in list_documents(conn):
                    if extract_text_for_doc(conn, row["doc_id"], Path(row["stored_path"]), row["media_type"]):
                        extracted += 1
                conn.close()

                built = []
                errors = []
                # Build image index if configured
                try:
                    _build_image_index(prefer_faiss=prefer_faiss)
                    built.append("image")
                except Exception as e:
                    errors.append(f"Image index failed: {str(e)}")
                # Build text index if possible
                try:
                    _build_text_index(prefer_faiss=prefer_faiss)
                    built.append("text")
                except Exception as e:
                    errors.append(f"Text index failed: {str(e)}")

                msg = f"Scanned {len(ingest_results)} files ({new_files} new, {existing_files} already indexed), extracted {extracted} docs, built: {', '.join(built) or 'none'}"
                if errors:
                    msg += f" | Errors: {'; '.join(errors)}"
                flash(msg, "success" if built else "error")
            except Exception as e:
                flash(str(e), "error")
        else:
            flash("Dataset path saved.", "success")

        return redirect(url_for("index"))

    @app.post("/search/text")
    def search_text_post():
        """Handle form submission and redirect to GET to avoid resubmission warning."""
        query = (request.form.get("query") or "").strip()
        modality = (request.form.get("modality") or "auto").strip().lower()
        top_k = int(request.form.get("top_k") or 5)

        if not query:
            return redirect(url_for("index"))

        # Save history
        conn = connect()
        add_search_history(
            conn,
            kind="text",
            query_text=query,
            target_modality=modality,
            top_k=top_k,
            uploaded_path=None,
        )
        conn.close()

        # Redirect to GET endpoint (PRG pattern)
        return redirect(url_for("search_text_get", q=query, modality=modality, top_k=top_k))

    @app.get("/search/text")
    def search_text_get():
        """Handle GET search - display results without form resubmission issues."""
        query = (request.args.get("q") or "").strip()
        modality = (request.args.get("modality") or "auto").strip().lower()
        top_k = int(request.args.get("top_k") or 5)
        prefer_faiss = True

        history = _get_history()

        if not query:
            cfg = load_config()
            return render_template(
                "index.html",
                text_results=None,
                image_results=None,
                error=None,
                message=None,
                query="",
                top_k=top_k,
                modality=modality,
                dataset_path=cfg.dataset_path,
                history=history,
                last_uploaded_filename=_last_uploaded_filename_from_history(history),
                **_index_status(),
            )

        try:
            if modality == "auto":
                text_results, image_results = auto_query(query=query, top_k=top_k, prefer_faiss=prefer_faiss)
            else:
                results = text_query(query=query, target_modality=modality, top_k=top_k, prefer_faiss=prefer_faiss)
                text_results = results
                image_results = None
        except FileNotFoundError as e:
            if modality == "text":
                msg = (
                    "Text index is missing. Run: fysearch build-index (text). "
                    "If you're mostly indexing images, OCR/text extraction may be empty; use Images (text→image) instead, "
                    "or set up OCR to generate searchable text."
                )
            elif modality == "auto":
                msg = "No indexes found. Run: fysearch build-index"
            else:
                msg = "Image index is missing. Run: fysearch build-index --modality image"
            return render_template(
                "index.html",
                text_results=None,
                image_results=None,
                error=msg,
                message=None,
                query=query,
                top_k=top_k,
                modality=modality,
                dataset_path=load_config().dataset_path,
                history=history,
                last_uploaded_filename=_last_uploaded_filename_from_history(history),
                **_index_status(),
            )
        except Exception as e:
            return render_template(
                "index.html",
                text_results=None,
                image_results=None,
                error=str(e),
                message=None,
                query=query,
                top_k=top_k,
                modality=modality,
                dataset_path=load_config().dataset_path,
                history=history,
                last_uploaded_filename=_last_uploaded_filename_from_history(history),
                **_index_status(),
            )

        return render_template(
            "index.html",
            text_results=text_results if text_results else None,
            image_results=image_results if image_results else None,
            error=None,
            message=None,
            query=query,
            top_k=top_k,
            modality=modality,
            dataset_path=load_config().dataset_path,
            history=history,
            last_uploaded_filename=_last_uploaded_filename_from_history(history),
            **_index_status(),
        )

    @app.post("/search/image")
    def search_image():
        top_k = int(request.form.get("top_k") or 5)
        prefer_faiss = True

        file = request.files.get("image")
        if file is None or file.filename == "":
            cfg = load_config()
            history = _get_history()
            return render_template(
                "index.html",
                text_results=None,
                image_results=None,
                error="No image uploaded",
                message=None,
                query="",
                top_k=top_k,
                modality="image",
                dataset_path=cfg.dataset_path,
                history=history,
                last_uploaded_filename=_last_uploaded_filename_from_history(history),
                **_index_status(),
            )

        paths = get_paths()
        upload_dir = paths.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save to a temporary file under data/uploads to make embedding backends happy.
        suffix = Path(file.filename).suffix or ".img"
        with tempfile.NamedTemporaryFile(delete=False, dir=upload_dir, suffix=suffix) as tmp:
            file.save(tmp)
            tmp_path = Path(tmp.name)

        uploaded_filename = tmp_path.name

        try:
            results = image_query(image_path=tmp_path, top_k=top_k, prefer_faiss=prefer_faiss)
        except FileNotFoundError:
            msg = "Image index is missing. Run: fysearch build-index --modality image"
            history = _get_history()
            return render_template(
                "index.html",
                text_results=None,
                image_results=None,
                error=msg,
                message=None,
                query="",
                top_k=top_k,
                modality="image",
                dataset_path=load_config().dataset_path,
                history=history,
                last_uploaded_filename=uploaded_filename,
                **_index_status(),
            )
        except Exception as e:
            history = _get_history()
            return render_template(
                "index.html",
                text_results=None,
                image_results=None,
                error=str(e),
                message=None,
                query="",
                top_k=top_k,
                modality="image",
                dataset_path=load_config().dataset_path,
                history=history,
                last_uploaded_filename=uploaded_filename,
                **_index_status(),
            )

        # Save history (store uploaded temp path; Clear will delete uploads)
        conn = connect()
        add_search_history(
            conn,
            kind="image",
            query_text=None,
            target_modality=None,
            top_k=top_k,
            uploaded_path=str(tmp_path),
        )
        conn.close()

        return render_template(
            "index.html",
            text_results=None,
            image_results=results,
            error=None,
            message=None,
            query="",
            top_k=top_k,
            modality="image",
            dataset_path=load_config().dataset_path,
            history=_get_history(),
            last_uploaded_filename=uploaded_filename,
            **_index_status(),
        )

    @app.get("/doc/<doc_id>")
    def serve_doc(doc_id: str):
        conn = connect()
        init_db(conn)
        row = conn.execute(
            "SELECT stored_path FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        conn.close()
        if not row:
            abort(404)
        path = Path(row["stored_path"]) 
        if not path.exists():
            abort(404)
        return send_file(path)

    return app
