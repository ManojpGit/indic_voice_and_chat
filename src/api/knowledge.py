"""Knowledge base / RAG endpoints (PRD §7.5).

A retriever instance is injected via ``set_retriever`` at app startup so the
endpoints stay testable in isolation. The retriever pairs a vector store
with an embedder and (optionally) a reranker — see ``src.rag.retriever``.

Endpoints:
- POST   /knowledge/ingest         multipart upload, parses + chunks + indexes
- GET    /knowledge/documents      list ingested documents (Phase 5+: paginate)
- DELETE /knowledge/documents/{id} remove an ingested document
- POST   /knowledge/query          retrieve top-k chunks for a query (debug)
- GET    /knowledge/stats          basic counts
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.auth import TenantContext, current_tenant
from src.interfaces.vector_store import Document
from src.rag.ingestion import ChunkConfig, detect_language, get_chunker, parse_document
from src.rag.retriever import HybridRetriever

log = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# --- DI -----------------------------------------------------------------


_retriever: Optional[HybridRetriever] = None
_chunk_config: ChunkConfig = ChunkConfig()
# In-memory document registry. Phase 5+ replaces this with the kb_documents
# Postgres table; for Phase 4 in-memory is sufficient and keeps tests fast.
_documents: dict[str, dict] = {}


def set_retriever(retriever: HybridRetriever, chunk_config: Optional[ChunkConfig] = None) -> None:
    global _retriever, _chunk_config, _documents
    _retriever = retriever
    if chunk_config is not None:
        _chunk_config = chunk_config
    _documents = {}


def _require_retriever() -> HybridRetriever:
    if _retriever is None:
        raise HTTPException(
            status_code=503,
            detail="knowledge base not initialized; set_retriever() not called",
        )
    return _retriever


# --- Schemas ------------------------------------------------------------


class DocumentInfo(BaseModel):
    id: str
    filename: str
    language: Optional[str] = None
    chunk_count: int


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    language: Optional[str]


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict] = None


class QueryHit(BaseModel):
    chunk_id: str
    content: str
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: dict


class QueryResponse(BaseModel):
    hits: list[QueryHit]
    total: int


class StatsResponse(BaseModel):
    document_count: int
    chunk_count: int


# --- Routes -------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    tenant: TenantContext = Depends(current_tenant),
) -> IngestResponse:
    retriever = _require_retriever()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    text = parse_document(file.filename or "uploaded", data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="document parsed to empty text")

    doc_id = document_id or _new_id()
    language = detect_language(text)
    chunker = get_chunker(_chunk_config)
    raw_chunks = chunker(text, {
        "filename": file.filename,
        "document_id": doc_id,
        "language": language,
        "tenant_id": tenant.id,
    })
    if not raw_chunks:
        raise HTTPException(status_code=400, detail="no chunks produced")

    docs = [
        Document(
            id=f"{doc_id}::chunk-{c.index}",
            content=c.text,
            metadata={
                **c.metadata,
                "section": c.index,
                "page": c.index,
            },
        )
        for c in raw_chunks
    ]
    indexed = await retriever.index(docs)
    _documents[doc_id] = {
        "tenant_id": tenant.id,
        "filename": file.filename,
        "language": language,
        "chunk_count": indexed,
        "chunk_ids": [d.id for d in docs],
    }
    log.info("ingested document", extra={"document_id": doc_id, "chunks": indexed})
    return IngestResponse(
        document_id=doc_id,
        filename=file.filename or "",
        chunks_indexed=indexed,
        language=language,
    )


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    tenant: TenantContext = Depends(current_tenant),
) -> DocumentsResponse:
    _require_retriever()
    items = [
        DocumentInfo(
            id=doc_id,
            filename=info["filename"] or doc_id,
            language=info.get("language"),
            chunk_count=info.get("chunk_count", 0),
        )
        for doc_id, info in _documents.items()
        if info.get("tenant_id") == tenant.id
    ]
    return DocumentsResponse(documents=items, total=len(items))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str, tenant: TenantContext = Depends(current_tenant),
) -> dict:
    retriever = _require_retriever()
    info = _documents.get(document_id)
    if info is None or info.get("tenant_id") != tenant.id:
        raise HTTPException(status_code=404, detail="document not found")
    _documents.pop(document_id, None)
    n = await retriever.delete(info["chunk_ids"])
    return {"document_id": document_id, "chunks_removed": n}


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest, tenant: TenantContext = Depends(current_tenant),
) -> QueryResponse:
    retriever = _require_retriever()
    # Always filter by tenant_id so query results never leak across tenants.
    filters = dict(req.filters or {})
    filters["tenant_id"] = tenant.id
    results = await retriever.search(req.query, top_k=req.top_k, filters=filters)
    hits = [
        QueryHit(
            chunk_id=r.document.id,
            content=r.document.content,
            score=r.score,
            dense_score=r.dense_score,
            bm25_score=r.bm25_score,
            rerank_score=r.rerank_score,
            metadata=r.document.metadata or {},
        )
        for r in results
    ]
    return QueryResponse(hits=hits, total=len(hits))


@router.get("/stats", response_model=StatsResponse)
async def stats(
    tenant: TenantContext = Depends(current_tenant),
) -> StatsResponse:
    _require_retriever()
    docs = [info for info in _documents.values() if info.get("tenant_id") == tenant.id]
    chunk_count = sum(info.get("chunk_count", 0) for info in docs)
    return StatsResponse(document_count=len(docs), chunk_count=chunk_count)


def _new_id() -> str:
    return f"doc_{uuid.uuid4().hex[:12]}"
