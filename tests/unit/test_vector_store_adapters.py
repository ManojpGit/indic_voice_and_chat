from __future__ import annotations

from pathlib import Path

import pytest

from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter


def _vec(d: int, fill: float) -> list[float]:
    return [fill] * d


@pytest.fixture
def store(tmp_faiss_index: str) -> FAISSAdapter:
    return FAISSAdapter({"embedding_dim": 4, "index_path": tmp_faiss_index})


@pytest.mark.asyncio
async def test_index_and_count(store: FAISSAdapter) -> None:
    docs = [
        Document(id="a", content="alpha", embedding=_vec(4, 1.0)),
        Document(id="b", content="beta", embedding=_vec(4, 0.5)),
    ]
    n = await store.index(docs)
    assert n == 2
    assert await store.count() == 2


@pytest.mark.asyncio
async def test_search_orders_by_cosine(store: FAISSAdapter) -> None:
    await store.index(
        [
            Document(id="a", content="close", embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(id="b", content="far", embedding=[0.0, 1.0, 0.0, 0.0]),
            Document(id="c", content="closer", embedding=[0.9, 0.1, 0.0, 0.0]),
        ]
    )
    results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
    ids = [r.document.id for r in results]
    assert ids[0] in ("a", "c")
    assert "b" == ids[-1]
    # Scores monotonically non-increasing
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_search_filter(store: FAISSAdapter) -> None:
    await store.index(
        [
            Document(id="a", content="x", metadata={"lang": "hi"}, embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(id="b", content="y", metadata={"lang": "en"}, embedding=[1.0, 0.0, 0.0, 0.0]),
        ]
    )
    results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=2, filters={"lang": "hi"})
    assert [r.document.id for r in results] == ["a"]


@pytest.mark.asyncio
async def test_index_rejects_missing_embedding(store: FAISSAdapter) -> None:
    with pytest.raises(ValueError):
        await store.index([Document(id="x", content="no vec", embedding=None)])


@pytest.mark.asyncio
async def test_index_rejects_dim_mismatch(store: FAISSAdapter) -> None:
    with pytest.raises(ValueError):
        await store.index([Document(id="x", content="bad", embedding=[1.0, 2.0])])


@pytest.mark.asyncio
async def test_delete_removes_documents(store: FAISSAdapter) -> None:
    await store.index(
        [
            Document(id="a", content="x", embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(id="b", content="y", embedding=[0.0, 1.0, 0.0, 0.0]),
            Document(id="c", content="z", embedding=[0.0, 0.0, 1.0, 0.0]),
        ]
    )
    removed = await store.delete(["b"])
    assert removed == 1
    assert await store.count() == 2
    results = await store.search([0.0, 1.0, 0.0, 0.0], top_k=3)
    assert "b" not in [r.document.id for r in results]


@pytest.mark.asyncio
async def test_persistence_round_trip(tmp_faiss_index: str) -> None:
    cfg = {"embedding_dim": 4, "index_path": tmp_faiss_index}
    store1 = FAISSAdapter(cfg)
    await store1.index(
        [Document(id="a", content="hi", embedding=[1.0, 0.0, 0.0, 0.0])]
    )
    assert Path(tmp_faiss_index).with_suffix(".faiss").exists()

    store2 = FAISSAdapter(cfg)
    assert await store2.count() == 1
    results = await store2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0].document.id == "a"
    assert results[0].document.content == "hi"


@pytest.mark.asyncio
async def test_search_empty_index(store: FAISSAdapter) -> None:
    assert await store.search([1.0, 0.0, 0.0, 0.0], top_k=5) == []
