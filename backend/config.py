from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    app_name: str = "insightly"
    app_version: str = "1.0.0"
    debug: bool = False

    # qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "insights"

    # cohere settings
    cohere_api_key: Optional[str] = None

    # openrouter api key
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "gpt-oss-120b"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # langfuse settings
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # upload
    upload_dir: str = "/data/uploads"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # additional settings
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5

settings = Settings()