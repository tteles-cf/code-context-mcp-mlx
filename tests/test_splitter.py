import os
import textwrap

import pytest

from local_code_context.splitter import (
    CHUNK_LINES,
    DEFAULT_EXTENSIONS,
    DEFAULT_IGNORE_DIRS,
    FileInfo,
    OVERLAP_LINES,
    make_chunk_id,
    scan_files,
    split_codebase,
    split_file,
)


@pytest.fixture
def sample_tree(tmp_path):
    """Create a small project tree for testing."""
    # Source files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main():\n    print('hello')\n")
    (src / "utils.ts").write_text(
        "export function add(a: number, b: number) { return a + b; }\n"
    )
    (src / "readme.txt").write_text("Not a code file\n")  # not in default extensions

    # Nested dir
    lib = src / "lib"
    lib.mkdir()
    (lib / "helper.js").write_text("function helper() {}\n")

    # Ignored dirs
    pycache = src / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-312.pyc").write_bytes(b"\x00\x00")

    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "dep.js").write_text("module.exports = {}\n")

    # Hidden dir
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.py").write_text("pass\n")

    return tmp_path


class TestScanFiles:
    def test_finds_source_files(self, sample_tree):
        files = scan_files(str(sample_tree))
        paths = {f.relative_path for f in files}
        assert "src/main.py" in paths
        assert "src/utils.ts" in paths
        assert "src/lib/helper.js" in paths

    def test_skips_non_code_extensions(self, sample_tree):
        files = scan_files(str(sample_tree))
        paths = {f.relative_path for f in files}
        assert "src/readme.txt" not in paths

    def test_skips_ignored_dirs(self, sample_tree):
        files = scan_files(str(sample_tree))
        paths = {f.relative_path for f in files}
        # __pycache__ and node_modules should be skipped
        for p in paths:
            assert "__pycache__" not in p
            assert "node_modules" not in p

    def test_skips_hidden_dirs(self, sample_tree):
        files = scan_files(str(sample_tree))
        paths = {f.relative_path for f in files}
        for p in paths:
            assert ".hidden" not in p

    def test_extra_extensions(self, sample_tree):
        files = scan_files(str(sample_tree), extra_extensions={".txt"})
        paths = {f.relative_path for f in files}
        assert "src/readme.txt" in paths

    def test_sorted_by_path(self, sample_tree):
        files = scan_files(str(sample_tree))
        paths = [f.relative_path for f in files]
        assert paths == sorted(paths)

    def test_content_hash_stable(self, sample_tree):
        files1 = scan_files(str(sample_tree))
        files2 = scan_files(str(sample_tree))
        for f1, f2 in zip(files1, files2):
            assert f1.content_hash == f2.content_hash

    def test_content_hash_changes(self, sample_tree):
        files1 = scan_files(str(sample_tree))
        h1 = {f.relative_path: f.content_hash for f in files1}

        (sample_tree / "src" / "main.py").write_text(
            "def main():\n    print('changed')\n"
        )

        files2 = scan_files(str(sample_tree))
        h2 = {f.relative_path: f.content_hash for f in files2}

        assert h1["src/main.py"] != h2["src/main.py"]
        assert h1["src/utils.ts"] == h2["src/utils.ts"]

    def test_empty_dir(self, tmp_path):
        files = scan_files(str(tmp_path))
        assert files == []


class TestSplitFile:
    def test_small_file_single_chunk(self, tmp_path):
        f = tmp_path / "small.py"
        f.write_text("line1\nline2\nline3\n")
        info = FileInfo(
            relative_path="small.py",
            absolute_path=str(f),
            file_extension=".py",
            content_hash="abc",
            size_bytes=18,
        )
        chunks = split_file(info)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3
        assert chunks[0].content == "line1\nline2\nline3\n"

    def test_large_file_multiple_chunks(self, tmp_path):
        f = tmp_path / "big.py"
        lines = [f"line {i}\n" for i in range(200)]
        f.write_text("".join(lines))
        info = FileInfo(
            relative_path="big.py",
            absolute_path=str(f),
            file_extension=".py",
            content_hash="abc",
            size_bytes=f.stat().st_size,
        )
        chunks = split_file(info)
        assert len(chunks) > 1

        # First chunk starts at line 1.
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == CHUNK_LINES

        # Chunks overlap.
        if len(chunks) >= 2:
            assert chunks[1].start_line == CHUNK_LINES - OVERLAP_LINES + 1

        # Last chunk ends at the final line.
        assert chunks[-1].end_line == 200

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("")
        info = FileInfo(
            relative_path="empty.py",
            absolute_path=str(f),
            file_extension=".py",
            content_hash="abc",
            size_bytes=0,
        )
        chunks = split_file(info)
        assert chunks == []

    def test_whitespace_only(self, tmp_path):
        f = tmp_path / "blank.py"
        f.write_text("   \n\n  \n")
        info = FileInfo(
            relative_path="blank.py",
            absolute_path=str(f),
            file_extension=".py",
            content_hash="abc",
            size_bytes=f.stat().st_size,
        )
        chunks = split_file(info)
        assert chunks == []

    def test_chunk_metadata(self, tmp_path):
        f = tmp_path / "code.ts"
        f.write_text("const x = 1;\n")
        info = FileInfo(
            relative_path="src/code.ts",
            absolute_path=str(f),
            file_extension=".ts",
            content_hash="abc",
            size_bytes=f.stat().st_size,
        )
        chunks = split_file(info)
        assert chunks[0].relative_path == "src/code.ts"
        assert chunks[0].file_extension == ".ts"


class TestMakeChunkId:
    def test_deterministic(self):
        id1 = make_chunk_id("src/main.py", 1, 80)
        id2 = make_chunk_id("src/main.py", 1, 80)
        assert id1 == id2

    def test_different_for_different_inputs(self):
        id1 = make_chunk_id("src/main.py", 1, 80)
        id2 = make_chunk_id("src/main.py", 81, 160)
        id3 = make_chunk_id("src/other.py", 1, 80)
        assert id1 != id2
        assert id1 != id3

    def test_length(self):
        chunk_id = make_chunk_id("file.py", 1, 10)
        assert len(chunk_id) == 16


class TestSplitCodebase:
    def test_full_pipeline(self, sample_tree):
        files, chunks = split_codebase(str(sample_tree))
        assert len(files) > 0
        assert len(chunks) > 0
        # Every chunk should have content.
        for c in chunks:
            assert c.content.strip()
