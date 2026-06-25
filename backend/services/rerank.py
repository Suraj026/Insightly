"""Re-ranks search results using Cohere's rerank API."""
from cohere import AsyncClientV2
from config import settings

class RerankService:
    """Re-scores search results by query relevance."""

    def __init__(self):
        self.client = AsyncClientV2(api_key=settings.cohere_api_key)
    
    async def rerank(self, query: str, results: list[dict], top_k: int = settings.rerank_top_k) -> list[dict]:
        """Re-ranks results and returns the top-k most relevant."""

        if not results:
            results = []        
        documents = [result["chunk_text"] for result in results]

        response = await self.client.rerank(
            model = "rerank-v3.5",
            query = query,
            documents = documents,
            top_n = top_k
        )
        reranked: list[dict] = []
        for item in response.results:
            original = results[item.index]
            reranked.append({
                "doc_id": original["doc_id"],
                "filename": original["filename"],
                "chunk_index": original["chunk_index"],
                "chunk_text": original["chunk_text"],
                "score": item.relevance_score
            })

        return reranked