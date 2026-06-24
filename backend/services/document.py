"""Handles file validation, text extraction, and storage."""
from uuid import uuid4
from io import BytesIO
from datetime import datetime
from pathlib import Path
from config import settings
from models.schema import DocumentInfo

class UnsupportedFileTypeError(Exception):
    """Raised when an unsupported file type is encountered."""
    pass

class FileTooLargeError(Exception):
    """Raised when the uploaded file exceeds the maximum allowed size."""
    pass

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024

class DocumentService:
    """Service for handling document operations."""

    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_file(filename: str, content: bytes) -> None:
        """Validates the file type and size."""
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {filename}"
            )
        if len(content) > MAX_FILE_SIZE:
            raise FileTooLargeError(
                f"File size exceeds the maximum limit of {MAX_FILE_SIZE / (1024 * 1024)} MB."
            )
        
    def extract_file(self, filename: str, content: bytes) -> str:
        """Extracts text from the file based on its type."""
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            import fitz
            import pymupdf4llm
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            file_content = pymupdf4llm.to_markdown(pdf_doc)
        elif ext == ".docx":
            import docx
            doc = docx.Document(BytesIO(content))
            file_content = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".txt":
            file_content = content.decode("utf-8")
        else:
            raise UnsupportedFileTypeError(
                f"Unsupported file type for extraction: {filename}"
            )
        
        return file_content
    
    def store_file(self, tenant_id: str, filename: str, content: bytes) -> tuple[str, Path]:
        """Stores the file in the upload directory and returns its unique ID and path."""
        doc_id = str(uuid4())
        file_path = self.upload_dir / tenant_id / f"{doc_id}_{filename}"

        with open(file_path, "wb") as f:
            f.write(content)
        return doc_id, file_path
    
    def _tenant_path(self, tenant_id: str) -> Path:
        """Returns the path for a specific tenant's uploads."""
        tenant_path = self.upload_dir / tenant_id
        return tenant_path
    
    def get_document_info(self, tenant_id: str, doc_id: str) -> DocumentInfo:
        """Retrieves metadata about the stored document."""
        tenant_path = self._tenant_path(tenant_id)
        if not tenant_path.exists():
            raise FileNotFoundError(
                "Tenant directory not found."
            )
        
        for file in tenant_path.iterdir():
            if file.name.startswith(doc_id):
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                return DocumentInfo(
                    doc_id=doc_id,
                    filename=file.name.split("_", 1)[1],
                    file_size=file.stat().st_size,
                    content_type=file.suffix,
                    uploaded_at=mtime,
                    chunk_count=None,
                )
        raise FileNotFoundError("Document not found.")
    
    def delete_document(self, tenant_id: str, doc_id: str) -> None:
        """Deletes the document with the given ID."""
        tenant_path = self._tenant_path(tenant_id)
        if not tenant_path.exists():
            raise FileNotFoundError(
                "Tenant directory not found."
            )
        
        for file in tenant_path.iterdir():
            if file.name.startswith(doc_id):
                file.unlink()
                return
        raise FileNotFoundError(
            "Document not found."
        )