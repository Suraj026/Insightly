"""Hybrid search service for combining vector and keyword search fused via RRF."""

from rank_bm25 import BM25Okapi
from config import settings
from services.embeddings import EmbeddingService
from services.vector_store import VectorStoreService

class HybridSearchService:
    """Service for performing hybrid search using vector and keyword search."""
    def __init__(self, embedder: EmbeddingService, vector_store: VectorStoreService):
        self.embedder = embedder
        self.vector_store = vector_store
        self._bm25: BM25Okapi = None
        self._all_chunks: list[dict] = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize the given text into a list of words."""
        return text.lower().split()

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add chunks to the BM25 index for keyword search."""
        self._all_chunks.extend(chunks)
        tokenized_corpus = [self._tokenize(chunk["chunk_text"]) for chunk in self._all_chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Keyword search via BM25."""
        if not self._bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked = sorted(
            [
                {**self._all_chunks[i], "score": float(scores[i])}
                for i in range(len(scores))
                if scores[i] > 0
            ],
            key=lambda x: x["score"],
            reverse=True,
        )
        return ranked[:top_k]
    
    def _rrf(self, results: list[list[dict]], k: int = settings.rrf_k) -> list[dict]:
        """Reciprocal Rank Fusion (RRF) to combine results from multiple search methods."""
        scores: dict[tuple[str, int], dict] = {}
        for result_set in results:
            for rank, item in enumerate(result_set):
                key = (item["doc_id"], item["chunk_index"])
                if key not in scores:
                    scores[key] = {k: v for k, v in item.items() if k != "score"}
                    scores[key]["score"] = 0.0
                scores[key]["score"] += 1.0 / (k + rank + 1)
        
        return sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    
    async def search(self, query: str) -> list[dict]:
        """"Runs hybrid search: vector + BM25, fused via RRF."""
        query_vector = await self.embedder.embed_query(query)
        vector_results = await self.vector_store.search(query_vector, settings.vector_top_k)
        bm25_results = self._bm25_search(query, settings.bm25_top_k)

        return self._rrf([vector_results, bm25_results])
        