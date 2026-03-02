"""Search orchestration — ties together embeddings, vector search, FTS, and RRF."""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import embed_query
from .rrf import RankedResult, rrf_fuse
from .store import Store


@dataclass
class SearchResult:
    id: str
    score: float
    content: str
    relative_path: str
    start_line: int
    end_line: int
    file_extension: str
    metadata: str


def _to_search_result(r: RankedResult) -> SearchResult:
    return SearchResult(
        id=r.id,
        score=r.score,
        content=r.chunk.content,
        relative_path=r.chunk.relative_path,
        start_line=r.chunk.start_line,
        end_line=r.chunk.end_line,
        file_extension=r.chunk.file_extension,
        metadata=r.chunk.metadata,
    )


def dense_search(
    store: Store,
    collection_name: str,
    query: str,
    limit: int = 10,
) -> list[SearchResult]:
    """Dense-only search: embed the query then KNN via sqlite-vec."""
    query_vector = embed_query(query)
    rows = store.vector_search(collection_name, query_vector, limit)
    return [
        SearchResult(
            id=id_,
            score=1.0 - distance,  # cosine distance -> similarity
            content=chunk.content,
            relative_path=chunk.relative_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            file_extension=chunk.file_extension,
            metadata=chunk.metadata,
        )
        for id_, distance, chunk in rows
    ]


def hybrid_search(
    store: Store,
    collection_name: str,
    query: str,
    limit: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Hybrid search: dense + sparse (FTS5), fused with RRF."""
    query_vector = embed_query(query)

    # Dense leg
    dense_rows = store.vector_search(collection_name, query_vector, limit)
    dense_results = [
        RankedResult(id=id_, score=distance, chunk=chunk)
        for id_, distance, chunk in dense_rows
    ]

    # Sparse leg
    sparse_rows = store.fts_search(collection_name, query, limit)
    sparse_results = [
        RankedResult(id=id_, score=score, chunk=chunk)
        for id_, score, chunk in sparse_rows
    ]

    fused = rrf_fuse([dense_results, sparse_results], k=rrf_k, limit=limit)
    return [_to_search_result(r) for r in fused]
