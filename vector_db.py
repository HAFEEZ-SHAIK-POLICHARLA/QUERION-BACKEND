"""
Qdrant Cloud storage layer.

Key changes from the original:
- QdrantClient is a module-level singleton — created once per backend process,
  not once per request. Matters once multiple users are querying concurrently.
- upsert() now stores a file_hash in the payload so duplicate PDFs can be
  detected before doing any (paid) embedding work.
- search() de-duplicates passages that appear identically across multiple
  sources, merging them into one context entry with a list of sources.
- clear_collection() lets the "Clear Knowledge Base" button actually work.
"""
from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger("uvicorn")

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Singleton QdrantClient — one connection reused across all requests."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if not url or not api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.")
        _client = QdrantClient(url=url, api_key=api_key, timeout=30)
        logger.info("QdrantClient initialized (singleton)")
    return _client


class QdrantStorage:
    def __init__(self, collection: str = "docs", dim: int = 3072):
        self.client = get_client()
        self.collection = collection
        self.dim = dim

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{self.collection}'")

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------
    def find_source_by_hash(self, file_hash: str) -> str | None:
        """Return the source filename already indexed under this hash, if any."""
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]
                ),
                limit=1,
            )
            if results:
                return results[0].payload.get("source")
        except Exception as exc:
            logger.warning(f"Duplicate check failed (continuing anyway): {exc}")
        return None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        if not ids:
            return
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def search(self, query_vector: list[float], top_k: int = 5) -> dict:
        """
        Search and de-duplicate: identical passages that exist under multiple
        sources are merged into one context entry with a combined source list.
        """
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )

        seen: dict[str, set[str]] = {}  # text -> set of sources
        order: list[str] = []

        for r in results.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if not text:
                continue
            if text not in seen:
                seen[text] = set()
                order.append(text)
            if source:
                seen[text].add(source)

        contexts = order
        sources_per_context = [sorted(seen[t]) for t in order]
        all_sources = sorted({s for srcs in sources_per_context for s in srcs})

        return {
            "contexts": contexts,
            "sources_per_context": sources_per_context,
            "sources": all_sources,
        }

    def list_sources(self) -> dict:
        """List all distinct source filenames currently indexed, with chunk counts."""
        sources: dict[str, int] = {}
        next_offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in results:
                src = (r.payload or {}).get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            if next_offset is None:
                break

        return {"sources": list(sources.keys()), "total_chunks": sum(sources.values())}

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def clear_collection(self) -> None:
        """Wipe the entire knowledge base (used by the 'Clear Knowledge Base' button)."""
        self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        logger.info(f"Cleared Qdrant collection '{self.collection}'")
