import os
from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    web_port: int = 8501
    debug: bool = True
    log_level: str = "INFO"
    secret_key: str = "dev-secret-key"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-3.5-turbo"
    
    # Hugging Face Configuration
    hf_token: Optional[str] = None
    
    # Database Configuration
    database_url: str = "sqlite:///./docqa.db"
    redis_url: str = "redis://localhost:6379"
    
    # Vector Database Configuration
    vector_db_type: str = "faiss"  # faiss, chroma, elasticsearch
    vector_db_path: str = "./vector_store"
    index_dimension: int = 1536
    
    # File Processing Configuration
    max_file_size_mb: int = 50
    allowed_file_types: str = "pdf,txt,docx"  # Will be parsed to list
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_query: int = 5
    
    # Search & Ranking Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "ms-marco-MiniLM-L-6-v2"
    search_top_k: int = 20
    rerank_top_k: int = 5
    
    # Performance Settings
    batch_size: int = 32
    index_refresh_interval: int = 300  # seconds
    cache_ttl: int = 3600  # seconds
    
    @field_validator('allowed_file_types', mode='after')
    @classmethod
    def parse_file_types(cls, v):
        if isinstance(v, str):
            # Comma-separated format
            return [ft.strip() for ft in v.split(',') if ft.strip()]
        return v
    
    def get_allowed_file_types(self) -> List[str]:
        """Get allowed file types as a list"""
        if isinstance(self.allowed_file_types, str):
            return [ft.strip() for ft in self.allowed_file_types.split(',') if ft.strip()]
        return self.allowed_file_types
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()