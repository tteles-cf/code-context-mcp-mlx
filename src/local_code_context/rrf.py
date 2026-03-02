"""Reciprocal Rank Fusion — merges multiple ranked lists into one."""

from __future__ import annotations

from dataclasses import dataclass

from .store import ChunkRow


@dataclass
class RankedResult:
    id: str
    score: float
    chunk: ChunkRow


def rrf_fuse(
    lists: list[list[RankedResult]],
    k: int = 60,
    limit: int | None = None,
) -> list[RankedResult]:
    """
    Fuse multiple ranked result lists using RRF.

    Args:
        lists: Arrays of results, each pre-sorted by their native ranking.
        k: RRF constant (default 60). Higher k reduces the impact of rank.
        limit: Max results to return. None for all.
    """
    scores: dict[str, tuple[float, ChunkRow]] = {}

    for result_list in lists:
        for rank, item in enumerate(result_list):
            rrf_score = 1.0 / (k + rank + 1)
            if item.id in scores:
                existing_score, existing_chunk = scores[item.id]
                scores[item.id] = (existing_score + rrf_score, existing_chunk)
            else:
                scores[item.id] = (rrf_score, item.chunk)

    results = [
        RankedResult(id=id_, score=score, chunk=chunk)
        for id_, (score, chunk) in scores.items()
    ]
    results.sort(key=lambda r: r.score, reverse=True)

    if limit is not None:
        results = results[:limit]
    return results
