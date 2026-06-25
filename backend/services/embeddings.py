"""Embedding service for generating text embeddings."""
from cohere import AsyncClientV2
from config import settings

class EmbeddingService:
    """Service for generating embeddings using Cohere API."""
    def __init__(self):
        self.client = AsyncClientV2(
            api_key=settings.cohere_api_key,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""
        response = await self.client.embed(
            inputs=texts,
            model="embed-v4.0",
            output_dimension=1024,
            input_type="search_document",
            embedding_types=["float"]
        )
        embeddings = response.embeddings.float_
        return embeddings
    
    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query."""
        response = await self.client.embed(
            inputs=[query],
            model="embed-v4.0",
            output_dimension=1024,
            input_type="search_query",
            embedding_types=["float"]
        )
        embeddings = response.embeddings.float 
        return embeddings