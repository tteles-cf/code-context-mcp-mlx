import pytest

from local_code_context.rrf import RankedResult, rrf_fuse
from local_code_context.store import ChunkRow


def _chunk(label: str) -> ChunkRow:
    return ChunkRow(
        id=label,
        content=label,
        relative_path=f"{label}.ts",
        start_line=1,
        end_line=1,
        file_extension=".ts",
        metadata="",
    )


class TestRRFFuse:
    def test_empty_input(self):
        assert rrf_fuse([]) == []

    def test_single_list_scoring(self):
        items = [
            RankedResult(id="a", score=0, chunk=_chunk("a")),
            RankedResult(id="b", score=0, chunk=_chunk("b")),
        ]
        result = rrf_fuse([items])
        assert len(result) == 2
        assert result[0].id == "a"
        assert result[0].score == pytest.approx(1 / 61)
        assert result[1].id == "b"
        assert result[1].score == pytest.approx(1 / 62)

    def test_overlapping_ids_sum_scores(self):
        list1 = [RankedResult(id="a", score=0, chunk=_chunk("a"))]
        list2 = [RankedResult(id="a", score=0, chunk=_chunk("a"))]
        result = rrf_fuse([list1, list2])
        assert len(result) == 1
        assert result[0].score == pytest.approx(1 / 61 + 1 / 61)

    def test_non_overlapping(self):
        list1 = [RankedResult(id="a", score=0, chunk=_chunk("a"))]
        list2 = [RankedResult(id="b", score=0, chunk=_chunk("b"))]
        result = rrf_fuse([list1, list2])
        assert len(result) == 2

    def test_respects_limit(self):
        items = [
            RankedResult(id="a", score=0, chunk=_chunk("a")),
            RankedResult(id="b", score=0, chunk=_chunk("b")),
            RankedResult(id="c", score=0, chunk=_chunk("c")),
        ]
        result = rrf_fuse([items], limit=2)
        assert len(result) == 2

    def test_sorts_descending(self):
        list1 = [
            RankedResult(id="low", score=0, chunk=_chunk("low")),
            RankedResult(id="high", score=0, chunk=_chunk("high")),
        ]
        list2 = [RankedResult(id="high", score=0, chunk=_chunk("high"))]
        result = rrf_fuse([list1, list2])
        assert result[0].id == "high"
