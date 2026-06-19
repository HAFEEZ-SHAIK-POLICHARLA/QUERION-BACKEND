    from fastapi import FastAPI, UploadFile, File
import logging
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: INGEST PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle= inngest.Throttle(
        limit = 2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit= inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id"
    )
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) ->RAGChunkAndSrc:
        pdf_path= ctx.event.data["pdf_path"]
        source_id= ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks=chunks_and_src.chunks
        source_id= chunks_and_src.source_id
        vecs= embed_texts(chunks)
        ids= [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}: {i}")) for i in range(len(chunks))]
        payloads= [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingessted=len(chunks))
     

    chunks_and_src= await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested= await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: QUERY PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int=5) -> RAGSearchResult:
        query_vec=embed_texts([question])[0]
        store=QdrantStorage()
        found= store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    question = ctx.event.data["question"]
    top_k =int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content =(
        "You are a retrieval-augmented assistant.\n\n"
    
    "Answer ONLY using the provided context.\n"
    "Do NOT use outside knowledge.\n"
    "Do NOT make assumptions.\n"
    "If the answer cannot be found in the context, respond with:\n"
    "'I could not find the answer in the provided documents.'\n\n"
    
    f"Context:\n{context_block}\n\n"
    
    f"Question:\n{question}\n\n"
    
    "Provide a precise answer based solely on the context."
    )

    adapter = ai.openai.Adapter(
    auth_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/auto",
    headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "RAG Production App"
    }
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body = {
            "max_tokens": 512,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}




app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_dir = Path(tempfile.gettempdir())
        pdf_path = temp_dir / file.filename

        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        chunks = load_and_chunk_pdf(str(pdf_path))
        vecs = embed_texts(chunks)

        source_id = file.filename

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids, vecs, payloads)

        return {
            "success": True,
            "chunks": len(chunks),
            "source": source_id
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/query")
async def query_pdf(payload: dict):
    try:
        question = payload.get("question")
        top_k = int(payload.get("top_k", 5))

        query_vec = embed_texts([question])[0]

        store = QdrantStorage()
        found = store.search(query_vec, top_k)

        context_block = "\n\n".join(
            f"- {c}" for c in found["contexts"]
        )

        user_content = (
            "You are a retrieval-augmented assistant.\n\n"
            "Answer ONLY using the provided context.\n"
            "Do NOT use outside knowledge.\n"
            "Do NOT make assumptions.\n"
            "If the answer cannot be found in the context, respond with:\n"
            "'I could not find the answer in the provided documents.'\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{question}\n\n"
            "Provide a precise answer based solely on the context."
        )

        adapter = ai.openai.Adapter(
            auth_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/auto",
            headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "RAG Production App"
            }
        )

        response = await adapter.chat_completions_create(
            messages=[
                {
                    "role": "system",
                    "content": "You answer using only the provided context."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            max_tokens=512,
            temperature=0.2
        )

        answer = response["choices"][0]["message"]["content"].strip()

        return {
            "answer": answer,
            "sources": found["sources"]
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


inngest.fast_api.serve(app, inngest_client,[rag_ingest_pdf, rag_query_pdf_ai])
