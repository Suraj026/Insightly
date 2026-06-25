"""Chunking service for splitting text into manageable pieces."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

class ChunkingService:
    """Service for chunking text into smaller pieces."""

    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators = ["\n\n", "\n", ".", " ", ""],
            length_function = len
        )

    def chunk_text(self, doc_id: str, filename: str, text: str) -> list[dict]:
        """Chunk the given text and return a list of dictionaries with chunked data."""
        chunks = self._splitter.split_text(text)
        return [{
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": idx,
            "chunk_text": chunk
        } for idx, chunk in enumerate(chunks)]
