import asyncio
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from loguru import logger

from .config import get_settings
from .database import create_tables, get_db, cache_manager
from .models import (
    DocumentResponse, QueryRequest, QueryResponse, FeedbackRequest,
    ChunkResponse
)
from .document_service import document_service
from .search_engine import search_engine
from .models import QueryLog

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Document Q&A Engine...")
    
    # Initialize database
    create_tables()
    logger.info("Database initialized")
    
    # Initialize search engine
    try:
        await search_engine.initialize()
        logger.info("Search engine initialized")
    except Exception as e:
        logger.error(f"Search engine initialization failed: {e}")
        raise
    
    logger.info("Application startup completed")
    
    yield
    
    # Cleanup
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Document Q&A Engine",
    description="A searchable document Q&A engine with embeddings and advanced ranking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = await search_engine.get_search_stats()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "search_engine": "ready",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Document management endpoints
@app.post("/documents", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a document"""
    try:
        logger.info(f"Uploading document: {file.filename}")
        result = await document_service.upload_document(file)
        
        logger.info(f"Document uploaded successfully: {result.filename}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all documents"""
    try:
        documents = await document_service.get_documents(skip=skip, limit=limit)
        return documents
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int):
    """Get specific document details"""
    try:
        document = await document_service.get_document(document_id)
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document"""
    try:
        success = await document_service.delete_document(document_id)
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    skip: int = 0,
    limit: int = 100
):
    """Get chunks for a document"""
    try:
        chunks = await document_service.get_document_chunks(
            document_id, skip=skip, limit=limit
        )
        return {"chunks": chunks, "total": len(chunks)}
        
    except Exception as e:
        logger.error(f"Failed to get chunks for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document chunks")


# Search endpoints
@app.post("/search", response_model=QueryResponse)
async def search_documents(request: QueryRequest):
    """Search documents with advanced query processing"""
    try:
        logger.info(f"Search query: {request.query}")
        
        # Perform search
        search_response = await search_engine.search(
            query=request.query,
            document_id=request.document_id,
            max_results=request.max_results,
            use_reranking=True
        )
        
        # Convert results to API format
        chunks = []
        for ranked_result in search_response.results:
            chunk = ChunkResponse(
                id=int(ranked_result.search_result.chunk_id.split('_')[-1]),  # Extract chunk index
                document_id=ranked_result.search_result.document_id,
                chunk_index=ranked_result.ranking_features.get('chunk_index', 0),
                content=ranked_result.search_result.content,
                token_count=ranked_result.ranking_features.get('token_count', 0),
                page_number=ranked_result.ranking_features.get('page_number'),
                relevance_score=ranked_result.relevance_score
            )
            chunks.append(chunk)
        
        # Generate answer from top chunks
        answer = await _generate_answer(request.query, chunks[:3])
        
        response = QueryResponse(
            query=request.query,
            answer=answer,
            relevant_chunks=chunks,
            search_time_ms=search_response.search_time_ms,
            total_time_ms=search_response.total_time_ms,
            document_count=len(set(chunk.document_id for chunk in chunks))
        )
        
        logger.info(
            f"Search completed: {len(chunks)} results in {search_response.total_time_ms:.1f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


async def _generate_answer(query: str, chunks: List[ChunkResponse]) -> str:
    """Generate answer from retrieved chunks"""
    if not chunks:
        return "I couldn't find any relevant information to answer your question."
    
    # Simple answer generation (can be enhanced with LLM integration)
    context_parts = []
    for chunk in chunks:
        snippet = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        context_parts.append(f"From document {chunk.document_id}: {snippet}")
    
    if len(context_parts) == 1:
        return f"Based on the document, {context_parts[0][context_parts[0].find(': ') + 2:]}"
    else:
        return f"Based on the available documents:\n\n" + "\n\n".join(context_parts)


@app.get("/search/suggestions")
async def get_query_suggestions(
    q: str,
    limit: int = 5
):
    """Get query suggestions"""
    try:
        suggestions = await search_engine.suggest_queries(q, limit)
        return {"suggestions": suggestions}
        
    except Exception as e:
        logger.error(f"Query suggestions failed: {e}")
        return {"suggestions": []}


# Feedback endpoint
@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """Submit feedback for query results"""
    try:
        query_log = db.query(QueryLog).filter(QueryLog.id == feedback.query_id).first()
        
        if not query_log:
            raise HTTPException(status_code=404, detail="Query not found")
        
        query_log.user_feedback = feedback.feedback
        db.commit()
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


# Analytics endpoints
@app.get("/analytics/stats")
async def get_analytics():
    """Get system analytics"""
    try:
        search_stats = await search_engine.get_search_stats()
        
        # Get query statistics from database
        with get_db() as db:
            total_queries = db.query(QueryLog).count()
            avg_search_time = db.query(QueryLog).with_entities(
                func.avg(QueryLog.search_time_ms)
            ).scalar() or 0
        
        return {
            "search_engine_stats": search_stats,
            "query_stats": {
                "total_queries": total_queries,
                "average_search_time_ms": float(avg_search_time)
            }
        }
        
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@app.get("/analytics/popular-queries")
async def get_popular_queries(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get most popular queries"""
    try:
        from sqlalchemy import func
        
        popular = db.query(
            QueryLog.query_text,
            func.count(QueryLog.query_text).label('count')
        ).group_by(
            QueryLog.query_text
        ).order_by(
            func.count(QueryLog.query_text).desc()
        ).limit(limit).all()
        
        return {
            "popular_queries": [
                {"query": query, "count": count}
                for query, count in popular
            ]
        }
        
    except Exception as e:
        logger.error(f"Popular queries failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve popular queries")


# Cache management
@app.post("/admin/clear-cache")
async def clear_cache():
    """Clear application cache"""
    try:
        await cache_manager.clear()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )