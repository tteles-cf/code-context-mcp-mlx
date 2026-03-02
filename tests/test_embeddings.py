import math

import pytest

from local_code_context.embeddings import (
    EMBEDDING_DIM,
    embed,
    embed_documents,
    embed_query,
)


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class TestEmbed:
    def test_correct_dimension(self):
        vecs = embed(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == EMBEDDING_DIM

    def test_normalised(self):
        [vec] = embed(["test normalisation"])
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_batch(self):
        vecs = embed(["one", "two", "three"])
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == EMBEDDING_DIM

    def test_empty_input(self):
        assert embed([]) == []

    def test_embed_query(self):
        vec = embed_query("What is TypeScript?")
        assert len(vec) == EMBEDDING_DIM

    def test_embed_documents(self):
        vecs = embed_documents(["doc one", "doc two"])
        assert len(vecs) == 2

    def test_similar_texts_score_higher(self):
        [q] = embed(["How to parse JSON in TypeScript"])
        [relevant] = embed(["function parseJSON(s: string) { return JSON.parse(s); }"])
        [irrelevant] = embed(["The weather in Paris is sunny today"])

        relevant_score = _cosine(q, relevant)
        irrelevant_score = _cosine(q, irrelevant)
        assert relevant_score > irrelevant_score
