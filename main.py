"""
Querion backend — FastAPI app.

Design notes:
- No Inngest. The original event-driven scaffolding is gone; everything
  runs as a simple FastAPI background task, which is enough for a single
  Render instance serving ~50 concurrent users.
- /upload never writes to disk. The file is held in memory only for the
  few seconds it takes to validate + hash + stream chunks out of it.
- Uploads are capped at 50MB (Render free tier has 512MB RAM total).
- Duplicate PDFs (same MD5 hash) are detected before any embedding calls
  are made, so re-uploading the same file is instant and free.
- /query accepts conversation history so follow-up questions get context,
  and caches identical questions in-memory for the life of the process.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from custom_types import JobStatus, QueryPayload
from data_loader import (
    InvalidPDFError,
    compute_file_hash,
    validate_pdf,
    iter_pdf_chunks,
    embed_texts,
    EMBED_BATCH_SIZE,
)
from vector_db import QdrantStorage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB hard limit

app = FastAPI(title="Querion Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores. Fine for a single Render instance; data resets on restart,
# which only means active upload jobs / cached queries are lost, not the
# indexed PDFs themselves (those live durably in Qdrant Cloud).
_jobs: dict[str, JobStatus] = {}
_query_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": "querion-backend"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Could not read upload: {exc}"})

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": f"File is {len(file_bytes) / (1024*1024):.1f}MB. "
                         f"The limit is 50MB — please upload a smaller PDF."
            },
        )

    try:
        page_count = validate_pdf(file_bytes)
    except InvalidPDFError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    file_hash = compute_file_hash(file_bytes)
    source_id = file.filename or f"document-{uuid.uuid4().hex[:8]}.pdf"

    store = QdrantStorage()
    existing_source = store.find_source_by_hash(file_hash)
    if existing_source:
        job_id = uuid.uuid4().hex
        _jobs[job_id] = JobStatus(
            job_id=job_id,
            status="duplicate",
            source=existing_source,
            total_pages=page_count,
            current_page=page_count,
            message=f"This PDF is already indexed as '{existing_source}'. Skipped re-processing.",
        )
        return {"job_id": job_id}

    job_id = uuid.uuid4().hex
    _jobs[job_id] = JobStatus(
        job_id=job_id,
        status="processing",
        source=source_id,
        total_pages=page_count,
        current_page=0,
        message="Starting...",
    )

    asyncio.create_task(_process_pdf_job(job_id, file_bytes, source_id, file_hash))

    return {"job_id": job_id}


async def _process_pdf_job(job_id: str, file_bytes: bytes, source_id: str, file_hash: str) -> None:
    job = _jobs[job_id]
    store = QdrantStorage()

    pending_chunks: list[str] = []
    chunks_indexed = 0

    def flush_batch(batch: list[str]) -> int:
        if not batch:
            return 0
        vectors = embed_texts(batch)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{file_hash}:{i}:{uuid.uuid4().hex[:6]}"))
               for i in range(len(batch))]
        payloads = [
            {"source": source_id, "text": batch[i], "file_hash": file_hash}
            for i in range(len(batch))
        ]
        store.upsert(ids, vectors, payloads)
        return len(batch)

    try:
        for current_page, total_pages, chunk in await asyncio.to_thread(
            lambda: list(iter_pdf_chunks(file_bytes))
        ):
            pending_chunks.append(chunk)
            job.current_page = current_page
            job.total_pages = total_pages
            job.message = f"Reading page {current_page} of {total_pages}..."

            if len(pending_chunks) >= EMBED_BATCH_SIZE:
                n = await asyncio.to_thread(flush_batch, pending_chunks)
                chunks_indexed += n
                job.chunks_indexed = chunks_indexed
                pending_chunks = []

        if pending_chunks:
            n = await asyncio.to_thread(flush_batch, pending_chunks)
            chunks_indexed += n
            job.chunks_indexed = chunks_indexed

        if chunks_indexed == 0:
            job.status = "error"
            job.error = "No readable text was found in this PDF (it may be scanned images without OCR text)."
            return

        job.status = "done"
        job.message = f"Indexed {chunks_indexed} passages from {job.total_pages} pages."

    except Exception as exc:
        logger.exception(f"Job {job_id} failed")
        job.status = "error"
        job.error = str(exc)


@app.get("/upload/status/{job_id}")
async def upload_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Unknown job_id"})
    return job.model_dump()


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
@app.get("/knowledge-base")
async def knowledge_base():
    store = QdrantStorage()
    info = await asyncio.to_thread(store.list_sources)
    return info


@app.post("/knowledge-base/clear")
async def clear_knowledge_base():
    store = QdrantStorage()
    await asyncio.to_thread(store.clear_collection)
    _query_cache.clear()
    return {"success": True, "message": "Knowledge base cleared."}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
@app.post("/query")
async def query_pdf(payload: QueryPayload):
    try:
        question = (payload.question or "").strip()
        top_k = max(1, min(int(payload.top_k or 5), 20))

        if not question:
            return JSONResponse(status_code=400, content={"error": "Question cannot be empty."})

        cache_key = f"{question.lower()}::{top_k}"
        if cache_key in _query_cache:
            cached = dict(_query_cache[cache_key])
            cached["cached"] = True
            return cached

        store = QdrantStorage()
        query_vec = (await asyncio.to_thread(embed_texts, [question]))
        if not query_vec:
            return JSONResponse(status_code=400, content={"error": "Could not embed the question."})

        found = await asyncio.to_thread(store.search, query_vec[0], top_k)

        if not found["contexts"]:
            answer = "I could not find the answer in the provided documents. Try uploading a PDF first, or rephrase your question."
            result = {"answer": answer, "sources": [], "num_contexts": 0, "cached": False}
            return result

        context_block = "\n\n".join(f"- {c}" for c in found["contexts"])

        history_block = ""
        if payload.history:
            recent = payload.history[-6:]  # last 3 exchanges
            history_lines = [f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in recent]
            history_block = "Previous conversation (for context on follow-up questions):\n" + "\n".join(history_lines) + "\n\n"

        user_content = (
            "You are a retrieval-augmented assistant.\n\n"
            "Answer ONLY using the provided context.\n"
            "Do NOT use outside knowledge.\n"
            "Do NOT make assumptions.\n"
            "If the answer cannot be found in the context, respond with:\n"
            "'I could not find the answer in the provided documents.'\n\n"
            f"{history_block}"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{question}\n\n"
            "Provide a precise answer based solely on the context."
        )

        client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

        response = await client.chat.completions.create(
            model="openrouter/auto",
            messages=[
                {"role": "system", "content": "You answer using only the provided context."},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
            temperature=0.2,
        )

        answer = (response.choices[0].message.content or "").strip()

        result = {
            "answer": answer,
            "sources": found["sources"],
            "num_contexts": len(found["contexts"]),
            "cached": False,
        }

        _query_cache[cache_key] = dict(result)
        if len(_query_cache) > 200:
            _query_cache.pop(next(iter(_query_cache)))

        return result

    except Exception as exc:
        logger.exception("Query failed")
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
