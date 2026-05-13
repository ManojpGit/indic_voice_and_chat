from __future__ import annotations

import math

from src.rag.embeddings import HashEmbedder, IdentityReranker


def test_hash_embedder_dim() -> None:
    e = HashEmbedder(dim=64)
    v = e.embed_query("hello")
    assert len(v) == 64


def test_hash_embedder_unit_norm() -> None:
    e = HashEmbedder(dim=128)
    for text in ["hello world", "नमस्ते दोस्तों", "x"]:
        v = e.embed_query(text)
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6


def test_hash_embedder_deterministic() -> None:
    e = HashEmbedder(dim=128)
    a = e.embed_query("hello world")
    b = e.embed_query("hello world")
    assert a == b


def test_hash_embedder_different_inputs_different_outputs() -> None:
    e = HashEmbedder(dim=128)
    a = e.embed_query("hello world")
    b = e.embed_query("completely different sentence")
    assert a != b


def test_hash_embedder_empty_string_is_anchor_vector() -> None:
    e = HashEmbedder(dim=8)
    v = e.embed_query("")
    assert v[0] == 1.0
    assert all(x == 0.0 for x in v[1:])


def test_hash_embedder_batch_matches_single() -> None:
    e = HashEmbedder(dim=64)
    batch = e.embed_documents(["a", "b", "c"])
    single = [e.embed_query(t) for t in ["a", "b", "c"]]
    assert batch == single


def test_identity_reranker_orders_by_overlap() -> None:
    r = IdentityReranker()
    query = "plan b data unlimited"
    docs = [
        "plan a has 100GB data",                  # some overlap
        "plan b has 500GB unlimited data",        # most overlap
        "completely unrelated text about cooking",  # no overlap
    ]
    scores = r.rerank(query, docs)
    assert scores[1] > scores[0] > scores[2]


def test_identity_reranker_empty_inputs() -> None:
    r = IdentityReranker()
    assert r.rerank("", ["x", "y"]) == [0.0, 0.0]
    assert r.rerank("hello", []) == []
