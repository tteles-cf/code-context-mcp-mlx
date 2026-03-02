"""File scanner and code splitter.

Scans a directory for source files and splits them into chunks suitable for
embedding and indexing. Uses line-based splitting with overlap to preserve
context across chunk boundaries.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

# Default file extensions to index.
DEFAULT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".scala",
        ".m",
        ".mm",
        ".md",
        ".markdown",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".sh",
        ".bash",
        ".zsh",
    }
)

# Directories to always skip.
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "target",  # rust/java
        ".egg-info",
    }
)

# Target chunk size in lines, and overlap.
CHUNK_LINES = 80
OVERLAP_LINES = 10


@dataclass
class FileChunk:
    """A chunk of source code from a single file."""

    relative_path: str
    file_extension: str
    content: str
    start_line: int
    end_line: int


@dataclass
class FileInfo:
    """Metadata about a scanned file."""

    relative_path: str
    absolute_path: str
    file_extension: str
    content_hash: str
    size_bytes: int


def scan_files(
    root: str,
    *,
    extra_extensions: set[str] | None = None,
    ignore_patterns: set[str] | None = None,
) -> list[FileInfo]:
    """Walk *root* and return metadata for all indexable files.

    Args:
        root: Absolute path to the codebase root.
        extra_extensions: Additional extensions to include beyond defaults.
        ignore_patterns: Additional directory names to skip.
    """
    extensions = DEFAULT_EXTENSIONS | (extra_extensions or set())
    ignore_dirs = DEFAULT_IGNORE_DIRS | (ignore_patterns or set())

    results: list[FileInfo] = []
    root = os.path.abspath(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place so os.walk skips them.
        dirnames[:] = [
            d for d in dirnames if d not in ignore_dirs and not d.startswith(".")
        ]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in extensions:
                continue

            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, root)

            try:
                raw = open(abs_path, "rb").read()
            except (OSError, PermissionError):
                continue

            content_hash = hashlib.md5(raw).hexdigest()
            results.append(
                FileInfo(
                    relative_path=rel_path,
                    absolute_path=abs_path,
                    file_extension=ext,
                    content_hash=content_hash,
                    size_bytes=len(raw),
                )
            )

    results.sort(key=lambda f: f.relative_path)
    return results


def split_file(file_info: FileInfo) -> list[FileChunk]:
    """Split a single file into overlapping line-based chunks.

    Returns an empty list if the file cannot be read or is empty.
    """
    try:
        text = open(
            file_info.absolute_path, "r", encoding="utf-8", errors="replace"
        ).read()
    except (OSError, PermissionError):
        return []

    if not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    total = len(lines)

    if total <= CHUNK_LINES:
        return [
            FileChunk(
                relative_path=file_info.relative_path,
                file_extension=file_info.file_extension,
                content=text,
                start_line=1,
                end_line=total,
            )
        ]

    chunks: list[FileChunk] = []
    start = 0

    while start < total:
        end = min(start + CHUNK_LINES, total)
        chunk_text = "".join(lines[start:end])
        chunks.append(
            FileChunk(
                relative_path=file_info.relative_path,
                file_extension=file_info.file_extension,
                content=chunk_text,
                start_line=start + 1,  # 1-indexed
                end_line=end,
            )
        )
        if end >= total:
            break
        start = end - OVERLAP_LINES

    return chunks


def make_chunk_id(relative_path: str, start_line: int, end_line: int) -> str:
    """Deterministic chunk ID from file path and line range."""
    raw = f"{relative_path}:{start_line}-{end_line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def split_codebase(
    root: str,
    *,
    extra_extensions: set[str] | None = None,
    ignore_patterns: set[str] | None = None,
) -> tuple[list[FileInfo], list[FileChunk]]:
    """Scan and split all files in a codebase.

    Returns (file_infos, chunks).
    """
    files = scan_files(
        root, extra_extensions=extra_extensions, ignore_patterns=ignore_patterns
    )
    all_chunks: list[FileChunk] = []
    for f in files:
        all_chunks.extend(split_file(f))
    return files, all_chunks
