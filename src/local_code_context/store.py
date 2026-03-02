"""
SQLite store using sqlite-vec for vector search + FTS5 for sparse search.

Schema:
  - `collections` — collection metadata
  - `vec_<name>` — vec0 virtual table for vector storage
  - `meta_<name>` — chunk metadata
  - `fts_<name>` — FTS5 virtual table for hybrid collections
"""

from __future__ import annotations

import re
import sqlite3
import struct
import threading
from dataclasses import dataclass

import sqlite_vec


@dataclass
class CollectionRow:
    name: str
    dimension: int
    is_hybrid: bool
    created_at: int


@dataclass
class ChunkRow:
    id: str
    content: str
    relative_path: str
    start_line: int
    end_line: int
    file_extension: str
    metadata: str


def to_safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


class Store:
    def __init__(self, db_path: str = ":memory:"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._migrate()

    def _migrate(self) -> None:
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                name       TEXT PRIMARY KEY,
                dimension  INTEGER NOT NULL,
                is_hybrid  INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT (unixepoch())
            )
        """)
        self.db.commit()

    # ── Collections ──────────────────────────────────────────────────

    def create_collection(self, name: str, dimension: int, is_hybrid: bool) -> None:
        safe = to_safe_name(name)
        row = self.db.execute(
            "SELECT name FROM collections WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            return  # idempotent

        self.db.execute(
            "INSERT INTO collections (name, dimension, is_hybrid) VALUES (?, ?, ?)",
            (name, dimension, 1 if is_hybrid else 0),
        )

        self.db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_{safe} USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{dimension}]
            )
        """)

        self.db.execute(f"""
            CREATE TABLE IF NOT EXISTS meta_{safe} (
                id            TEXT PRIMARY KEY,
                content       TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                start_line    INTEGER NOT NULL,
                end_line      INTEGER NOT NULL,
                file_extension TEXT NOT NULL,
                metadata      TEXT NOT NULL DEFAULT ''
            )
        """)

        if is_hybrid:
            self.db.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_{safe} USING fts5(
                    id UNINDEXED,
                    content,
                    relative_path UNINDEXED,
                    start_line UNINDEXED,
                    end_line UNINDEXED,
                    file_extension UNINDEXED,
                    metadata UNINDEXED
                )
            """)

        self.db.commit()

    def drop_collection(self, name: str) -> None:
        safe = to_safe_name(name)
        self.db.execute(f"DROP TABLE IF EXISTS vec_{safe}")
        self.db.execute(f"DROP TABLE IF EXISTS meta_{safe}")
        self.db.execute(f"DROP TABLE IF EXISTS fts_{safe}")
        self.db.execute("DELETE FROM collections WHERE name = ?", (name,))
        self.db.commit()

    def has_collection(self, name: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM collections WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def list_collections(self) -> list[str]:
        rows = self.db.execute("SELECT name FROM collections ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def get_collection(self, name: str) -> CollectionRow | None:
        row = self.db.execute(
            "SELECT name, dimension, is_hybrid, created_at FROM collections WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return CollectionRow(
            name=row[0],
            dimension=row[1],
            is_hybrid=bool(row[2]),
            created_at=row[3],
        )

    # ── Entities ─────────────────────────────────────────────────────

    def insert_chunks(
        self,
        collection_name: str,
        chunks: list[tuple[str, list[float], ChunkRow]],
    ) -> None:
        """Insert chunks as (id, vector, chunk_data) tuples."""
        col = self.get_collection(collection_name)
        if col is None:
            raise ValueError(f"Collection not found: {collection_name}")

        safe = to_safe_name(collection_name)

        for chunk_id, vector, chunk in chunks:
            self.db.execute(
                f"INSERT OR REPLACE INTO vec_{safe} (id, embedding) VALUES (?, ?)",
                (chunk_id, _serialize_f32(vector)),
            )
            self.db.execute(
                f"INSERT OR REPLACE INTO meta_{safe} "
                "(id, content, relative_path, start_line, end_line, file_extension, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    chunk.content,
                    chunk.relative_path,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.file_extension,
                    chunk.metadata,
                ),
            )
            if col.is_hybrid:
                self.db.execute(
                    f"INSERT OR REPLACE INTO fts_{safe} "
                    "(id, content, relative_path, start_line, end_line, file_extension, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk_id,
                        chunk.content,
                        chunk.relative_path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.file_extension,
                        chunk.metadata,
                    ),
                )

        self.db.commit()

    def delete_by_ids(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        col = self.get_collection(collection_name)
        if col is None:
            return

        safe = to_safe_name(collection_name)
        placeholders = ",".join("?" for _ in ids)

        self.db.execute(f"DELETE FROM vec_{safe} WHERE id IN ({placeholders})", ids)
        self.db.execute(f"DELETE FROM meta_{safe} WHERE id IN ({placeholders})", ids)
        if col.is_hybrid:
            self.db.execute(f"DELETE FROM fts_{safe} WHERE id IN ({placeholders})", ids)
        self.db.commit()

    def delete_by_path(self, collection_name: str, relative_path: str) -> None:
        col = self.get_collection(collection_name)
        if col is None:
            return
        safe = to_safe_name(collection_name)

        rows = self.db.execute(
            f"SELECT id FROM meta_{safe} WHERE relative_path = ?",
            (relative_path,),
        ).fetchall()
        ids = [r[0] for r in rows]
        self.delete_by_ids(collection_name, ids)

    # ── Search ───────────────────────────────────────────────────────

    def vector_search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
    ) -> list[tuple[str, float, ChunkRow]]:
        """Dense KNN search. Returns (id, distance, chunk) sorted by distance ascending."""
        safe = to_safe_name(collection_name)
        rows = self.db.execute(
            f"""
            SELECT v.id, v.distance,
                   m.content, m.relative_path, m.start_line, m.end_line,
                   m.file_extension, m.metadata
            FROM vec_{safe} v
            JOIN meta_{safe} m ON v.id = m.id
            WHERE v.embedding MATCH ?
            AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_f32(query_vector), limit),
        ).fetchall()

        return [
            (
                r[0],
                r[1],
                ChunkRow(
                    id=r[0],
                    content=r[2],
                    relative_path=r[3],
                    start_line=r[4],
                    end_line=r[5],
                    file_extension=r[6],
                    metadata=r[7],
                ),
            )
            for r in rows
        ]

    def fts_search(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 10,
    ) -> list[tuple[str, float, ChunkRow]]:
        """Sparse FTS5 search. Returns (id, score, chunk) sorted by relevance."""
        col = self.get_collection(collection_name)
        if col is None or not col.is_hybrid:
            return []

        safe = to_safe_name(collection_name)

        # Sanitise: strip non-word/non-space, split, discard short tokens
        tokens = [t for t in re.sub(r"[^\w\s]", "", query_text).split() if len(t) > 1]
        if not tokens:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        try:
            rows = self.db.execute(
                f"""
                SELECT id, content, relative_path, start_line, end_line,
                       file_extension, metadata, -rank AS score
                FROM fts_{safe}
                WHERE fts_{safe} MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            (
                r[0],
                r[7],
                ChunkRow(
                    id=r[0],
                    content=r[1],
                    relative_path=r[2],
                    start_line=r[3],
                    end_line=r[4],
                    file_extension=r[5],
                    metadata=r[6],
                ),
            )
            for r in rows
        ]

    def query_by_path(
        self,
        collection_name: str,
        relative_path: str,
        limit: int = 16384,
    ) -> list[ChunkRow]:
        safe = to_safe_name(collection_name)
        try:
            rows = self.db.execute(
                f"""
                SELECT id, content, relative_path, start_line, end_line,
                       file_extension, metadata
                FROM meta_{safe}
                WHERE relative_path = ?
                LIMIT ?
                """,
                (relative_path, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            ChunkRow(
                id=r[0],
                content=r[1],
                relative_path=r[2],
                start_line=r[3],
                end_line=r[4],
                file_extension=r[5],
                metadata=r[6],
            )
            for r in rows
        ]

    def close(self) -> None:
        self.db.close()
