from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from pydantic import BaseModel

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    content_hash = Column(String(64), unique=True, index=True)
    upload_time = Column(DateTime, default=func.now())
    processed = Column(Boolean, default=False)
    processing_time = Column(Float, nullable=True)
    total_chunks = Column(Integer, default=0)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    queries = relationship("QueryLog", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), index=True)
    token_count = Column(Integer, nullable=False)
    embedding_id = Column(String(100), nullable=True)  # Vector DB ID
    page_number = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    response_text = Column(Text, nullable=True)
    relevant_chunks = Column(JSON, nullable=True)  # List of chunk IDs
    search_time_ms = Column(Float, nullable=False)
    total_time_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    user_feedback = Column(String(20), nullable=True)  # thumbs_up, thumbs_down
    
    # Relationships
    document = relationship("Document", back_populates="queries")


# Pydantic Models for API
class DocumentBase(BaseModel):
    filename: str
    file_type: str
    file_size: int


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: int
    original_filename: str
    upload_time: datetime
    processed: bool
    total_chunks: int
    processing_time: Optional[float] = None
    
    class Config:
        from_attributes = True


class ChunkResponse(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    token_count: int
    page_number: Optional[int] = None
    relevance_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    query: str
    document_id: Optional[int] = None
    max_results: int = 5
    include_metadata: bool = False


class QueryResponse(BaseModel):
    query: str
    answer: str
    relevant_chunks: List[ChunkResponse]
    search_time_ms: float
    total_time_ms: float
    document_count: int


class FeedbackRequest(BaseModel):
    query_id: int
    feedback: str  # thumbs_up, thumbs_down