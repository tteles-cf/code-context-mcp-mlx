import json
import os

import pytest

from local_code_context.snapshot import (
    CodebaseSnapshot,
    IndexStatus,
    SnapshotStore,
    collection_name_for_path,
)


@pytest.fixture
def snap_store(tmp_path):
    """SnapshotStore backed by a temp file."""
    return SnapshotStore(snapshot_file=str(tmp_path / "snapshots.json"))


class TestCollectionNameForPath:
    def test_deterministic(self):
        assert collection_name_for_path("/foo/bar") == collection_name_for_path(
            "/foo/bar"
        )

    def test_different_paths_differ(self):
        assert collection_name_for_path("/foo") != collection_name_for_path("/bar")

    def test_prefix(self):
        name = collection_name_for_path("/some/path")
        assert name.startswith("codebase_")

    def test_length(self):
        name = collection_name_for_path("/some/path")
        # "codebase_" (9) + 12 hex chars = 21
        assert len(name) == 21


class TestSnapshotStore:
    def test_get_nonexistent(self, snap_store):
        assert snap_store.get("/nonexistent") is None

    def test_start_indexing(self, snap_store):
        snap = snap_store.start_indexing("/my/project", total_files=10, total_chunks=50)
        assert snap.status == IndexStatus.INDEXING
        assert snap.total_files == 10
        assert snap.total_chunks == 50
        assert snap.started_at > 0

    def test_get_after_start(self, snap_store):
        snap_store.start_indexing("/my/project", 10, 50)
        snap = snap_store.get("/my/project")
        assert snap is not None
        assert snap.status == IndexStatus.INDEXING

    def test_update_progress(self, snap_store):
        snap_store.start_indexing("/my/project", 10, 50)
        snap_store.update_progress("/my/project", indexed_files=3, indexed_chunks=15)
        snap = snap_store.get("/my/project")
        assert snap.indexed_files == 3
        assert snap.indexed_chunks == 15
        assert snap.progress_pct == 30.0

    def test_finish_indexing(self, snap_store):
        snap_store.start_indexing("/my/project", 10, 50)
        snap_store.finish_indexing(
            "/my/project",
            total_files=10,
            total_chunks=50,
            file_hashes={"main.py": "abc123", "utils.py": "def456"},
        )
        snap = snap_store.get("/my/project")
        assert snap.status == IndexStatus.INDEXED
        assert snap.progress_pct == 100.0
        assert snap.total_files == 10
        assert snap.total_chunks == 50
        assert snap.file_hashes == {"main.py": "abc123", "utils.py": "def456"}
        assert snap.finished_at > snap.started_at

    def test_fail_indexing(self, snap_store):
        snap_store.start_indexing("/my/project", 10, 50)
        snap_store.fail_indexing("/my/project", "Out of memory")
        snap = snap_store.get("/my/project")
        assert snap.status == IndexStatus.FAILED
        assert snap.error == "Out of memory"

    def test_remove(self, snap_store):
        snap_store.start_indexing("/my/project", 10, 50)
        snap_store.remove("/my/project")
        assert snap_store.get("/my/project") is None

    def test_remove_nonexistent(self, snap_store):
        snap_store.remove("/nonexistent")  # should not raise

    def test_list_all(self, snap_store):
        snap_store.start_indexing("/project1", 5, 20)
        snap_store.start_indexing("/project2", 8, 30)
        all_snaps = snap_store.list_all()
        assert len(all_snaps) == 2

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "snapshots.json")

        # Write some data.
        store1 = SnapshotStore(snapshot_file=path)
        store1.start_indexing("/my/project", 10, 50)
        store1.finish_indexing("/my/project", 10, 50, {"a.py": "hash1"})

        # Read it back in a new instance.
        store2 = SnapshotStore(snapshot_file=path)
        snap = store2.get("/my/project")
        assert snap is not None
        assert snap.status == IndexStatus.INDEXED
        assert snap.file_hashes == {"a.py": "hash1"}

    def test_corrupted_file(self, tmp_path):
        path = str(tmp_path / "snapshots.json")
        with open(path, "w") as f:
            f.write("not valid json{{{")

        store = SnapshotStore(snapshot_file=path)
        assert store.list_all() == []

    def test_update_progress_nonexistent(self, snap_store):
        # Should not raise.
        snap_store.update_progress("/nonexistent", 1, 5)

    def test_fail_nonexistent(self, snap_store):
        snap_store.fail_indexing("/nonexistent", "error")  # should not raise

    def test_finish_nonexistent(self, snap_store):
        snap_store.finish_indexing("/nonexistent", 0, 0, {})  # should not raise
