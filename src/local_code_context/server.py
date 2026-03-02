"""
MCP server for local code context — codebase-oriented API.

Point at a directory, and the server handles scanning, splitting, embedding,
indexing, and hybrid search automatically.

Tools:
  - index_codebase(path, force)       — index a codebase directory
  - search_code(path, query, limit)   — hybrid search over an indexed codebase
  - clear_index(path)                 — drop the index for a codebase
  - get_indexing_status(path)          — check indexing progress
"""

from __future__ import annotations

import os
import threading

from mcp.server.fastmcp import FastMCP

from .embeddings import EMBEDDING_DIM, embed_documents, _get_model
from .search import dense_search, hybrid_search
from .snapshot import IndexStatus, SnapshotStore, collection_name_for_path
from .splitter import make_chunk_id, scan_files, split_file
from .store import ChunkRow, Store

DB_PATH = os.environ.get("CODE_CONTEXT_DB", "code-context.db")

store = Store(DB_PATH)
snapshots = SnapshotStore()
mcp = FastMCP("local-code-context")

# Warm the embedding model in a background thread so the MCP server can
# start accepting connections immediately while the model loads.
threading.Thread(target=_get_model, daemon=True).start()

# Track active indexing threads so we don't double-index.
_indexing_locks: dict[str, threading.Lock] = {}
_indexing_locks_lock = threading.Lock()


def _get_indexing_lock(path: str) -> threading.Lock:
    """Get or create a per-codebase lock."""
    key = os.path.abspath(path)
    with _indexing_locks_lock:
        if key not in _indexing_locks:
            _indexing_locks[key] = threading.Lock()
        return _indexing_locks[key]


def _prepare_index(path: str, force: bool) -> dict | None:
    """Synchronous preparation: scan files, split chunks, set up collection.

    Returns a context dict for the background worker, or None if there's
    nothing to do (e.g. no files found).
    """
    abs_path = os.path.abspath(path)
    col_name = collection_name_for_path(abs_path)

    # Scan all files.
    files = scan_files(abs_path)
    if not files:
        return None

    # Determine which files need (re-)indexing.
    existing = snapshots.get(abs_path)
    old_hashes = existing.file_hashes if existing and not force else {}

    files_to_index = []
    for f in files:
        if (
            f.relative_path in old_hashes
            and old_hashes[f.relative_path] == f.content_hash
        ):
            continue
        files_to_index.append(f)

    # Split files into chunks.
    all_chunks = []
    for f in files_to_index:
        all_chunks.extend(split_file(f))

    # If force, recreate the collection. Otherwise ensure it exists.
    if force and store.has_collection(col_name):
        store.drop_collection(col_name)
    if not store.has_collection(col_name):
        store.create_collection(col_name, EMBEDDING_DIM, is_hybrid=True)

    # For incremental: delete chunks for files that changed.
    if not force:
        for f in files_to_index:
            store.delete_by_path(col_name, f.relative_path)

        # Also delete chunks for files that no longer exist.
        current_paths = {f.relative_path for f in files}
        if old_hashes:
            removed = set(old_hashes.keys()) - current_paths
            for rp in removed:
                store.delete_by_path(col_name, rp)

    # Set status to INDEXING immediately (before background thread).
    snapshots.start_indexing(abs_path, len(files), len(all_chunks))

    return {
        "abs_path": abs_path,
        "col_name": col_name,
        "files": files,
        "all_chunks": all_chunks,
        "force": force,
        "existing": existing,
    }


def _do_index(ctx: dict) -> None:
    """Background indexing worker — embeds and inserts chunks."""
    abs_path = ctx["abs_path"]
    col_name = ctx["col_name"]
    files = ctx["files"]
    all_chunks = ctx["all_chunks"]
    force = ctx["force"]
    existing = ctx["existing"]

    lock = _get_indexing_lock(abs_path)
    if not lock.acquire(blocking=False):
        return  # Already indexing this codebase.

    try:
        # Index in batches.
        BATCH_SIZE = 32
        indexed_chunks = 0
        indexed_files_set: set[str] = set()

        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            texts = [c.content for c in batch]
            vectors = embed_documents(texts)

            entries = []
            for chunk, vec in zip(batch, vectors):
                chunk_id = make_chunk_id(
                    chunk.relative_path, chunk.start_line, chunk.end_line
                )
                row = ChunkRow(
                    id=chunk_id,
                    content=chunk.content,
                    relative_path=chunk.relative_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    file_extension=chunk.file_extension,
                    metadata="",
                )
                entries.append((chunk_id, vec, row))
                indexed_files_set.add(chunk.relative_path)

            store.insert_chunks(col_name, entries)
            indexed_chunks += len(batch)
            snapshots.update_progress(abs_path, len(indexed_files_set), indexed_chunks)

        # Build final file hashes (unchanged + newly indexed).
        final_hashes = {f.relative_path: f.content_hash for f in files}

        # For incremental indexing, total_chunks includes unchanged chunks.
        if not force and existing:
            # Count chunks that were already indexed for unchanged files.
            changed_file_count = len(all_chunks)
            prev_total = existing.total_chunks
            total_chunks = indexed_chunks + max(0, prev_total - changed_file_count)
        else:
            total_chunks = indexed_chunks

        snapshots.finish_indexing(
            abs_path,
            total_files=len(files),
            total_chunks=total_chunks,
            file_hashes=final_hashes,
        )

    except Exception as e:
        snapshots.fail_indexing(abs_path, str(e))
    finally:
        lock.release()


# ── Tools ────────────────────────────────────────────────────────────


@mcp.tool()
def index_codebase(
    path: str,
    force: bool = False,
) -> str:
    """Index a codebase directory for code search.

    Scans for source files, splits them into chunks, embeds them, and stores
    them for hybrid search. Runs in the background — use get_indexing_status
    to check progress.

    Incremental by default: only re-indexes files that changed since the last
    run. Use force=True to re-index everything from scratch.

    Args:
        path: Absolute path to the codebase directory.
        force: Force full re-index even if already indexed. Default False.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        return f'Error: "{abs_path}" is not a directory.'

    # Check if already indexing.
    existing = snapshots.get(abs_path)
    if existing and existing.status == IndexStatus.INDEXING:
        return (
            f'Indexing already in progress for "{abs_path}" '
            f"({existing.progress_pct}% complete)."
        )

    # Synchronous phase: scan, split, set status to INDEXING.
    ctx = _prepare_index(abs_path, force)
    if ctx is None:
        return f'No indexable source files found in "{abs_path}".'

    n_chunks = len(ctx["all_chunks"])
    n_files = len(ctx["files"])

    if n_chunks == 0:
        # All files unchanged — nothing to embed.
        snapshots.finish_indexing(
            abs_path,
            total_files=n_files,
            total_chunks=existing.total_chunks if existing else 0,
            file_hashes={f.relative_path: f.content_hash for f in ctx["files"]},
        )
        return (
            f'All {n_files} files in "{abs_path}" are unchanged. '
            f"Index is up to date ({existing.total_chunks if existing else 0} chunks)."
        )

    # Async phase: embed and insert in background.
    threading.Thread(target=_do_index, args=(ctx,), daemon=True).start()

    if existing and existing.status == IndexStatus.INDEXED and not force:
        return (
            f'Incremental re-index started for "{abs_path}": '
            f"{n_chunks} chunks from changed files will be re-indexed. "
            f"Use get_indexing_status to check progress."
        )
    return (
        f'Indexing started for "{abs_path}": '
        f"{n_files} files, {n_chunks} chunks to embed. "
        f"Use get_indexing_status to check progress."
    )


@mcp.tool()
def search_code(
    path: str,
    query: str,
    limit: int = 10,
) -> str:
    """Search for code in an indexed codebase.

    Uses hybrid search (vector similarity + full-text) with Reciprocal Rank
    Fusion for best results. The codebase must be indexed first with
    index_codebase.

    Args:
        path: Absolute path to the codebase directory.
        query: Natural language search query.
        limit: Max results (1-50, default 10).
    """
    abs_path = os.path.abspath(path)
    col_name = collection_name_for_path(abs_path)

    if not store.has_collection(col_name):
        snap = snapshots.get(abs_path)
        if snap and snap.status == IndexStatus.INDEXING:
            return (
                f'Codebase "{abs_path}" is still being indexed '
                f"({snap.progress_pct}% complete). Try again shortly."
            )
        return f'Codebase "{abs_path}" has not been indexed. Run index_codebase first.'

    limit = max(1, min(limit, 50))

    col = store.get_collection(col_name)
    assert col is not None

    results = (
        hybrid_search(store, col_name, query, limit)
        if col.is_hybrid
        else dense_search(store, col_name, query, limit)
    )

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        ext = r.file_extension.lstrip(".")
        lines.append(
            f"### {i}. {r.relative_path}:{r.start_line}-{r.end_line} "
            f"(score: {r.score:.4f})\n```{ext}\n{r.content}\n```"
        )
    return "\n\n".join(lines)


@mcp.tool()
def clear_index(path: str) -> str:
    """Remove the index for a codebase directory.

    Drops all indexed chunks and removes tracking state. The codebase can
    be re-indexed afterwards with index_codebase.

    Args:
        path: Absolute path to the codebase directory.
    """
    abs_path = os.path.abspath(path)
    col_name = collection_name_for_path(abs_path)

    if store.has_collection(col_name):
        store.drop_collection(col_name)

    snapshots.remove(abs_path)
    return f'Index cleared for "{abs_path}".'


@mcp.tool()
def get_indexing_status(path: str) -> str:
    """Check the indexing status of a codebase directory.

    Returns the current state: not_found, indexing (with progress %),
    indexed (with stats), or failed (with error message).

    Args:
        path: Absolute path to the codebase directory.
    """
    abs_path = os.path.abspath(path)
    snap = snapshots.get(abs_path)

    if snap is None:
        return f'No index found for "{abs_path}". Run index_codebase to index it.'

    if snap.status == IndexStatus.INDEXING:
        return (
            f'Indexing in progress for "{abs_path}": '
            f"{snap.progress_pct}% complete "
            f"({snap.indexed_files}/{snap.total_files} files, "
            f"{snap.indexed_chunks}/{snap.total_chunks} chunks)."
        )

    if snap.status == IndexStatus.INDEXED:
        elapsed = snap.finished_at - snap.started_at
        return (
            f'Indexed "{abs_path}": '
            f"{snap.total_files} files, {snap.total_chunks} chunks "
            f"(took {elapsed:.1f}s)."
        )

    if snap.status == IndexStatus.FAILED:
        return (
            f'Indexing failed for "{abs_path}": {snap.error} '
            f"({snap.progress_pct}% complete when failed)."
        )

    return f'Status for "{abs_path}": {snap.status.value}.'


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
