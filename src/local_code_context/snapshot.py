"""Indexing snapshot persistence.

Tracks the indexing state of each codebase path. Persists to
~/.local-code-context/snapshots.json so state survives server restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum


SNAPSHOT_DIR = os.path.expanduser("~/.local-code-context")
SNAPSHOT_FILE = os.path.join(SNAPSHOT_DIR, "snapshots.json")


class IndexStatus(str, Enum):
    NOT_FOUND = "not_found"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass
class FileState:
    """Hash of a file at the time it was last indexed."""

    relative_path: str
    content_hash: str


@dataclass
class CodebaseSnapshot:
    """Persistent state for one indexed codebase."""

    path: str
    collection_name: str
    status: IndexStatus = IndexStatus.NOT_FOUND
    total_files: int = 0
    total_chunks: int = 0
    indexed_files: int = 0
    indexed_chunks: int = 0
    progress_pct: float = 0.0
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    file_hashes: dict[str, str] = field(default_factory=dict)  # rel_path -> hash


def collection_name_for_path(path: str) -> str:
    """Deterministic collection name from an absolute codebase path."""
    abs_path = os.path.abspath(path)
    digest = hashlib.md5(abs_path.encode()).hexdigest()[:12]
    return f"codebase_{digest}"


class SnapshotStore:
    """Thread-safe, disk-persisted snapshot store."""

    def __init__(self, snapshot_file: str = SNAPSHOT_FILE):
        self._file = snapshot_file
        self._lock = threading.Lock()
        self._snapshots: dict[str, CodebaseSnapshot] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file, "r") as f:
                data = json.load(f)
            for key, val in data.items():
                val["status"] = IndexStatus(val["status"])
                self._snapshots[key] = CodebaseSnapshot(**val)
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            # Corrupted file — start fresh.
            self._snapshots = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        data = {}
        for key, snap in self._snapshots.items():
            d = asdict(snap)
            d["status"] = snap.status.value
            data[key] = d
        with open(self._file, "w") as f:
            json.dump(data, f, indent=2)

    def _key(self, path: str) -> str:
        return os.path.abspath(path)

    def get(self, path: str) -> CodebaseSnapshot | None:
        with self._lock:
            return self._snapshots.get(self._key(path))

    def start_indexing(
        self, path: str, total_files: int, total_chunks: int
    ) -> CodebaseSnapshot:
        key = self._key(path)
        with self._lock:
            snap = CodebaseSnapshot(
                path=key,
                collection_name=collection_name_for_path(path),
                status=IndexStatus.INDEXING,
                total_files=total_files,
                total_chunks=total_chunks,
                started_at=time.time(),
            )
            self._snapshots[key] = snap
            self._save()
            return snap

    def update_progress(
        self, path: str, indexed_files: int, indexed_chunks: int
    ) -> None:
        key = self._key(path)
        with self._lock:
            snap = self._snapshots.get(key)
            if snap is None:
                return
            snap.indexed_files = indexed_files
            snap.indexed_chunks = indexed_chunks
            snap.progress_pct = (
                round(indexed_chunks / snap.total_chunks * 100, 1)
                if snap.total_chunks > 0
                else 0.0
            )
            self._save()

    def finish_indexing(
        self,
        path: str,
        total_files: int,
        total_chunks: int,
        file_hashes: dict[str, str],
    ) -> None:
        key = self._key(path)
        with self._lock:
            snap = self._snapshots.get(key)
            if snap is None:
                return
            snap.status = IndexStatus.INDEXED
            snap.total_files = total_files
            snap.total_chunks = total_chunks
            snap.indexed_files = total_files
            snap.indexed_chunks = total_chunks
            snap.progress_pct = 100.0
            snap.finished_at = time.time()
            snap.file_hashes = file_hashes
            snap.error = ""
            self._save()

    def fail_indexing(self, path: str, error: str) -> None:
        key = self._key(path)
        with self._lock:
            snap = self._snapshots.get(key)
            if snap is None:
                return
            snap.status = IndexStatus.FAILED
            snap.error = error
            snap.finished_at = time.time()
            self._save()

    def remove(self, path: str) -> None:
        key = self._key(path)
        with self._lock:
            self._snapshots.pop(key, None)
            self._save()

    def list_all(self) -> list[CodebaseSnapshot]:
        with self._lock:
            return list(self._snapshots.values())
