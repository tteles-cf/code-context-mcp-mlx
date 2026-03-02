import pytest

from local_code_context.store import ChunkRow, Store, to_safe_name


class TestToSafeName:
    def test_replaces_special_chars(self):
        assert to_safe_name("my-collection.v2") == "my_collection_v2"

    def test_leaves_clean_names(self):
        assert to_safe_name("my_collection") == "my_collection"


class TestCollections:
    def setup_method(self):
        self.store = Store(":memory:")

    def test_create_and_list(self):
        self.store.create_collection("test", 1024, False)
        assert self.store.list_collections() == ["test"]

    def test_has_collection(self):
        assert not self.store.has_collection("test")
        self.store.create_collection("test", 1024, False)
        assert self.store.has_collection("test")

    def test_create_idempotent(self):
        self.store.create_collection("test", 1024, False)
        self.store.create_collection("test", 1024, False)
        assert self.store.list_collections() == ["test"]

    def test_drop(self):
        self.store.create_collection("test", 1024, True)
        self.store.drop_collection("test")
        assert not self.store.has_collection("test")

    def test_drop_idempotent(self):
        self.store.drop_collection("nonexistent")  # should not raise

    def test_get_collection(self):
        self.store.create_collection("test", 1024, True)
        col = self.store.get_collection("test")
        assert col is not None
        assert col.name == "test"
        assert col.dimension == 1024
        assert col.is_hybrid is True

    def test_hybrid_creates_fts(self):
        self.store.create_collection("hybrid", 4, True)
        self.store.db.execute(
            "INSERT INTO fts_hybrid (id, content, relative_path, start_line, end_line, file_extension, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("id1", "hello world", "test.ts", 1, 10, ".ts", ""),
        )
        row = self.store.db.execute(
            "SELECT id FROM fts_hybrid WHERE fts_hybrid MATCH ?", ("hello",)
        ).fetchone()
        assert row is not None
        assert row[0] == "id1"


def _chunk(id_: str, content: str, path: str = "a.ts") -> ChunkRow:
    return ChunkRow(
        id=id_,
        content=content,
        relative_path=path,
        start_line=1,
        end_line=1,
        file_extension=".ts",
        metadata="",
    )


class TestChunks:
    def setup_method(self):
        self.store = Store(":memory:")
        self.store.create_collection("test", 4, True)

    def test_insert_and_query_by_path(self):
        self.store.insert_chunks(
            "test",
            [("c1", [1, 0, 0, 0], _chunk("c1", "function hello() {}", "src/index.ts"))],
        )
        results = self.store.query_by_path("test", "src/index.ts")
        assert len(results) == 1
        assert results[0].id == "c1"
        assert results[0].content == "function hello() {}"

    def test_delete_by_ids(self):
        self.store.insert_chunks(
            "test",
            [
                ("c1", [1, 0, 0, 0], _chunk("c1", "hello", "a.ts")),
                ("c2", [0, 1, 0, 0], _chunk("c2", "world", "b.ts")),
            ],
        )
        self.store.delete_by_ids("test", ["c1"])
        assert self.store.query_by_path("test", "a.ts") == []
        assert len(self.store.query_by_path("test", "b.ts")) == 1

    def test_delete_by_path(self):
        self.store.insert_chunks(
            "test",
            [
                ("c1", [1, 0, 0, 0], _chunk("c1", "hello", "src/a.ts")),
                ("c2", [0, 1, 0, 0], _chunk("c2", "world", "src/a.ts")),
                ("c3", [0, 0, 1, 0], _chunk("c3", "keep", "src/b.ts")),
            ],
        )
        self.store.delete_by_path("test", "src/a.ts")
        assert self.store.query_by_path("test", "src/a.ts") == []
        assert len(self.store.query_by_path("test", "src/b.ts")) == 1

    def test_insert_nonexistent_collection_raises(self):
        with pytest.raises(ValueError, match="Collection not found"):
            self.store.insert_chunks(
                "nope",
                [("c1", [1, 0, 0, 0], _chunk("c1", "x", "x"))],
            )


class TestVectorSearch:
    def setup_method(self):
        self.store = Store(":memory:")
        self.store.create_collection("test", 4, False)
        self.store.insert_chunks(
            "test",
            [
                ("c1", [1, 0, 0, 0], _chunk("c1", "alpha", "a.ts")),
                ("c2", [0, 1, 0, 0], _chunk("c2", "beta", "b.ts")),
                ("c3", [0, 0, 1, 0], _chunk("c3", "gamma", "c.ts")),
            ],
        )

    def test_returns_nearest(self):
        results = self.store.vector_search("test", [0.9, 0.1, 0, 0], 2)
        assert len(results) == 2
        assert results[0][0] == "c1"

    def test_respects_limit(self):
        results = self.store.vector_search("test", [1, 0, 0, 0], 1)
        assert len(results) == 1


class TestFTSSearch:
    def setup_method(self):
        self.store = Store(":memory:")
        self.store.create_collection("hybrid", 4, True)
        self.store.insert_chunks(
            "hybrid",
            [
                (
                    "c1",
                    [1, 0, 0, 0],
                    _chunk(
                        "c1",
                        "function parseJSON(input: string) { return JSON.parse(input); }",
                        "parser.ts",
                    ),
                ),
                (
                    "c2",
                    [0, 1, 0, 0],
                    _chunk(
                        "c2",
                        "class DatabaseConnection { connect() {} }",
                        "db.ts",
                    ),
                ),
            ],
        )

    def test_finds_matching(self):
        results = self.store.fts_search("hybrid", "parseJSON function", 10)
        assert len(results) >= 1
        assert results[0][0] == "c1"

    def test_empty_for_non_hybrid(self):
        self.store.create_collection("dense_only", 4, False)
        assert self.store.fts_search("dense_only", "hello", 10) == []

    def test_garbage_query(self):
        assert self.store.fts_search("hybrid", "!@#$%", 10) == []
