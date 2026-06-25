"""Vector store service for managing vector embeddings."""
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from config import settings
from services.embeddings import EmbeddingService

class VectorStoreService:
    """Service for managing vector embeddings in Qdrant."""
    def __init__(self, embedder: EmbeddingService):
        # using :memory: for prototyping, use cloud for production
        if settings.qdrant_url == "http://localhost:6333":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(
                url = settings.qdrant_url,
                api_key = settings.qdrant_api_key
            )
        self.embedder = embedder
        self.collection_name = settings.qdrant_collection
    
    def get_collection(self):
        """Get the collection from Qdrant, create if it doesn't exist."""
        try:
            self.client.collection_exists(collection_name=self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                )
            )
    
    def index_document(self, chunks: list[dict]) -> int:
        """Index the given document chunks into Qdrant."""
        texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)

        points = [
            models.PointStruct(
                id=hash(f"{chunk['doc_id']}_{chunk['chunk_index']}"),
                vector=embeddings,
                payload={
                    "doc_id": chunk["doc_id"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["chunk_text"]
                }
            )
            for chunk, embeddings in zip(chunks, embeddings)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)
    
    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Search for the top_k most similar documents to the given query vector."""
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        return [{
            "doc_id": hit.payload["doc_id"],
            "filename": hit.payload["filename"],
            "chunk_index": hit.payload["chunk_index"],
            "chunk_text": hit.payload["chunk_text"],
            "score": hit.score
        } for hit in result.points]