from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

# A single cited chunk
class Citation(BaseModel):
    text: str = Field(description="The text of the cited chunk.")
    score: float = Field(description="The score of the cited chunk.")
    doc_id: str = Field(description="The document ID of the cited chunk.")
    chunk_index: int = Field(description="The index of the chunk in the document.")

# User's query
class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="The user's query to be answered.")

# Answer with citations
class QueryResponse(BaseModel):
    trace_id: str = Field(description="The trace ID for the request.")
    answer: str = Field(description="The answer to the user's query.")
    citations: list[Citation] = Field(description="A list of cited chunks that support the answer.")

# Upload result
class UploadResponse(BaseModel):
    doc_id: str = Field(description="The document ID of the uploaded document.")
    filename: str = Field(description="The filename of the uploaded document.")
    chunk_count: int = Field(description="The number of chunks created from the uploaded document.")
    status: str = Field(description="The status of the upload operation.")

# Document metadata
class DocumentInfo(BaseModel):
    doc_id: str = Field(description="The document ID of the uploaded document.")
    filename: str = Field(description="The filename of the uploaded document.")
    file_size: int = Field(description="The size of the uploaded document.")
    content_type: str = Field(description="The content type of the uploaded document.")
    chunk_count: int = Field(description="The number of chunks created from the uploaded document.")
    uploaded_at: datetime = Field(description="The timestamp when the document was uploaded.")

# Error payload
class ErrorResponse(BaseModel):
    trace_id: Optional[str] = Field(description="The trace ID for the request, if available.")
    detail: str = Field(description="A description of the error.")

# Health check shape
class HealthResponse(BaseModel):
    status: str = Field(description="The health status of the application.")
    app: str = Field(description="The name of the application.")