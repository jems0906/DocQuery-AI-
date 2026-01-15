import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from loguru import logger
from .config import get_settings
from .database import get_db_session
from .models import Document, DocumentChunk, DocumentCreate, DocumentResponse
from .document_processor import DocumentProcessor
from .search_engine import search_engine

settings = get_settings()


class DocumentService:
    """Service for managing documents and their processing"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
    async def upload_document(self, file: UploadFile) -> DocumentResponse:
        """Upload and process a document"""
        temp_path = None
        
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            # Save uploaded file temporarily
            temp_path = self.upload_dir / f"temp_{file.filename}"
            
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Validate file
            is_valid, error_msg = self.processor.validate_file(str(temp_path), file.filename)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)
            
            # Process document
            processed_doc = await self.processor.process_file(str(temp_path), file.filename)
            
            # Check for existing document with same content hash
            with get_db_session() as db:
                existing_doc = db.query(Document).filter(
                    Document.content_hash == processed_doc.content_hash
                ).first()
                
                if existing_doc:
                    logger.info(f"Document already exists: {existing_doc.filename}")
                    return DocumentResponse.from_orm(existing_doc)
            
            # Save document to permanent location
            permanent_path = self.upload_dir / f"{processed_doc.content_hash}_{file.filename}"
            shutil.move(str(temp_path), str(permanent_path))
            
            # Save to database
            doc_id = await self._save_document(processed_doc, str(permanent_path))
            
            # Add to search index
            await self._index_document(doc_id, processed_doc.chunks)
            
            # Get saved document
            with get_db_session() as db:
                saved_doc = db.query(Document).filter(Document.id == doc_id).first()
                return DocumentResponse.from_orm(saved_doc)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
        finally:
            # Cleanup temporary file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
    
    async def _save_document(self, processed_doc, file_path: str) -> int:
        """Save processed document to database"""
        with get_db_session() as db:
            # Create document record
            doc = Document(
                filename=processed_doc.filename,
                original_filename=processed_doc.filename,
                file_type=processed_doc.file_type,
                file_size=processed_doc.file_size,
                file_path=file_path,
                content_hash=processed_doc.content_hash,
                processed=True,
                processing_time=processed_doc.processing_time,
                total_chunks=len(processed_doc.chunks),
                metadata=processed_doc.metadata
            )
            
            db.add(doc)
            db.commit()
            db.refresh(doc)
            
            # Create chunk records
            for i, chunk in enumerate(processed_doc.chunks):
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk.content,
                    content_hash=self._hash_content(chunk.content),
                    token_count=len(chunk.content.split()),
                    page_number=chunk.page_number,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata=chunk.metadata
                )
                db.add(db_chunk)
            
            db.commit()
            logger.info(f"Saved document {doc.id} with {len(processed_doc.chunks)} chunks")
            return doc.id
    
    async def _index_document(self, doc_id: int, chunks: List) -> bool:
        """Add document chunks to search index"""
        try:
            chunk_data = [
                {
                    'content': chunk.content,
                    'page_number': chunk.page_number,
                    'chunk_index': i,
                    'token_count': len(chunk.content.split()),
                    'metadata': chunk.metadata or {}
                }
                for i, chunk in enumerate(chunks)
            ]
            
            success = await search_engine.add_document(doc_id, chunk_data)
            
            if success:
                logger.info(f"Document {doc_id} indexed successfully")
            else:
                logger.error(f"Failed to index document {doc_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return False
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content"""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def get_documents(self, skip: int = 0, limit: int = 100) -> List[DocumentResponse]:
        """Get list of documents"""
        try:
            with get_db_session() as db:
                documents = db.query(Document).offset(skip).limit(limit).all()
                return [DocumentResponse.from_orm(doc) for doc in documents]
                
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve documents")
    
    async def get_document(self, document_id: int) -> DocumentResponse:
        """Get specific document"""
        try:
            with get_db_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                
                if not document:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                return DocumentResponse.from_orm(document)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve document")
    
    async def delete_document(self, document_id: int) -> bool:
        """Delete document and remove from index"""
        try:
            with get_db_session() as db:
                document = db.query(Document).filter(Document.id == document_id).first()
                
                if not document:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                # Remove from search index
                await search_engine.remove_document(document_id)
                
                # Delete file
                try:
                    if os.path.exists(document.file_path):
                        os.unlink(document.file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete file {document.file_path}: {e}")
                
                # Delete from database (cascades to chunks)
                db.delete(document)
                db.commit()
                
                logger.info(f"Deleted document {document_id}")
                return True
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete document")
    
    async def get_document_chunks(self, document_id: int, skip: int = 0, 
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get chunks for a document"""
        try:
            with get_db_session() as db:
                chunks = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id
                ).offset(skip).limit(limit).all()
                
                return [
                    {
                        'id': chunk.id,
                        'chunk_index': chunk.chunk_index,
                        'content': chunk.content,
                        'token_count': chunk.token_count,
                        'page_number': chunk.page_number,
                        'metadata': chunk.metadata
                    }
                    for chunk in chunks
                ]
                
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve document chunks")


# Global document service instance
document_service = DocumentService()