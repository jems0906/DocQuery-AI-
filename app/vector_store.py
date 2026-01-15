import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import faiss
from pathlib import Path
from loguru import logger
from .config import get_settings

settings = get_settings()


@dataclass
class SearchResult:
    chunk_id: str
    document_id: int
    content: str
    score: float
    metadata: Dict[str, Any] = None


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_vectors(self, vectors: List[List[float]], chunk_ids: List[str], 
                         metadata: List[Dict[str, Any]]) -> bool:
        pass
    
    @abstractmethod
    async def search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: int) -> bool:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store for high-performance similarity search"""
    
    def __init__(self, dimension: int, store_path: str):
        self.dimension = dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.store_path / "faiss.index"
        self.metadata_file = self.store_path / "metadata.json"
        
        self.index = None
        self.metadata = {}  # chunk_id -> metadata mapping
        self.chunk_to_idx = {}  # chunk_id -> faiss index mapping
        self.idx_to_chunk = {}  # faiss index -> chunk_id mapping
        
        self._load_index()
    
    def _load_index(self):
        """Load existing index or create new one"""
        try:
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index with Inner Product (cosine similarity)
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"Created new FAISS index (dimension: {self.dimension})")
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", {})
                    self.chunk_to_idx = data.get("chunk_to_idx", {})
                    self.idx_to_chunk = {v: k for k, v in self.chunk_to_idx.items()}
                logger.info(f"Loaded metadata for {len(self.metadata)} chunks")
                
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.index = faiss.IndexFlatIP(self.dimension)
    
    def _save_index(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, str(self.index_file))
            
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    "metadata": self.metadata,
                    "chunk_to_idx": self.chunk_to_idx
                }, f, indent=2)
            
            logger.debug(f"Saved FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    async def add_vectors(self, vectors: List[List[float]], chunk_ids: List[str], 
                         metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to the index"""
        try:
            if len(vectors) != len(chunk_ids) or len(vectors) != len(metadata):
                raise ValueError("Vectors, chunk_ids, and metadata must have same length")
            
            # Convert to numpy array and normalize for cosine similarity
            np_vectors = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(np_vectors)
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(np_vectors)
            
            # Update mappings
            for i, chunk_id in enumerate(chunk_ids):
                idx = start_idx + i
                self.chunk_to_idx[chunk_id] = idx
                self.idx_to_chunk[idx] = chunk_id
                self.metadata[chunk_id] = metadata[i]
            
            self._save_index()
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS: {e}")
            return False
    
    async def search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        """Search for similar vectors"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query vector
            query_np = np.array([query_vector], dtype=np.float32)
            faiss.normalize_L2(query_np)
            
            # Search
            scores, indices = self.index.search(query_np, min(top_k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                chunk_id = self.idx_to_chunk.get(idx)
                if chunk_id and chunk_id in self.metadata:
                    metadata = self.metadata[chunk_id]
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        document_id=metadata.get("document_id"),
                        content=metadata.get("content", ""),
                        score=float(score),
                        metadata=metadata
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def delete_document(self, document_id: int) -> bool:
        """Delete all chunks for a document (requires full rebuild)"""
        try:
            # Find chunks to delete
            chunks_to_delete = [
                chunk_id for chunk_id, meta in self.metadata.items()
                if meta.get("document_id") == document_id
            ]
            
            if not chunks_to_delete:
                return True
            
            # Remove from metadata
            for chunk_id in chunks_to_delete:
                self.metadata.pop(chunk_id, None)
                self.chunk_to_idx.pop(chunk_id, None)
            
            # Rebuild index (FAISS doesn't support efficient deletion)
            await self._rebuild_index()
            
            logger.info(f"Deleted {len(chunks_to_delete)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document from FAISS: {e}")
            return False
    
    async def _rebuild_index(self):
        """Rebuild index after deletion"""
        try:
            # Create new index
            new_index = faiss.IndexFlatIP(self.dimension)
            new_chunk_to_idx = {}
            new_idx_to_chunk = {}
            
            # Get all remaining vectors
            vectors = []
            chunk_ids = []
            
            for chunk_id in self.metadata.keys():
                if "embedding" in self.metadata[chunk_id]:
                    vectors.append(self.metadata[chunk_id]["embedding"])
                    chunk_ids.append(chunk_id)
            
            if vectors:
                # Add vectors to new index
                np_vectors = np.array(vectors, dtype=np.float32)
                faiss.normalize_L2(np_vectors)
                new_index.add(np_vectors)
                
                # Update mappings
                for i, chunk_id in enumerate(chunk_ids):
                    new_chunk_to_idx[chunk_id] = i
                    new_idx_to_chunk[i] = chunk_id
            
            # Replace old index
            self.index = new_index
            self.chunk_to_idx = new_chunk_to_idx
            self.idx_to_chunk = new_idx_to_chunk
            
            self._save_index()
            logger.info(f"Rebuilt FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP",
            "total_chunks": len(self.metadata),
            "total_documents": len(set(
                meta.get("document_id") for meta in self.metadata.values()
                if meta.get("document_id") is not None
            ))
        }


class VectorStoreManager:
    """Manager for vector store operations"""
    
    def __init__(self, embedding_dimension: int = None):
        self.dimension = embedding_dimension or settings.index_dimension
        self.store_type = settings.vector_db_type
        self.store_path = settings.vector_db_path
        
        self.vector_store = self._create_vector_store()
    
    def _create_vector_store(self) -> BaseVectorStore:
        """Create vector store based on configuration"""
        if self.store_type.lower() == "faiss":
            return FAISSVectorStore(self.dimension, self.store_path)
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    async def add_document_chunks(self, document_id: int, chunks: List[Dict[str, Any]], 
                                 embeddings: List[List[float]]) -> bool:
        """Add document chunks with embeddings to vector store"""
        try:
            chunk_ids = [f"doc_{document_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Prepare metadata
            metadata = []
            for i, chunk in enumerate(chunks):
                meta = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "content": chunk.get("content", ""),
                    "page_number": chunk.get("page_number"),
                    "token_count": chunk.get("token_count", 0),
                    "embedding": embeddings[i]  # Store embedding for rebuilding
                }
                metadata.append(meta)
            
            return await self.vector_store.add_vectors(embeddings, chunk_ids, metadata)
            
        except Exception as e:
            logger.error(f"Failed to add document chunks to vector store: {e}")
            return False
    
    async def search_similar(self, query_vector: List[float], 
                           top_k: int = 10, document_id: int = None) -> List[SearchResult]:
        """Search for similar chunks"""
        try:
            results = await self.vector_store.search(query_vector, top_k * 2)  # Get more for filtering
            
            # Filter by document if specified
            if document_id is not None:
                results = [r for r in results if r.document_id == document_id]
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def delete_document(self, document_id: int) -> bool:
        """Delete all chunks for a document"""
        return await self.vector_store.delete_document(document_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return await self.vector_store.get_stats()