"""Simplified API for testing the basic setup"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from typing import List, Dict, Any

app = FastAPI(
    title="Document Q&A Engine - Simple Version",
    description="Basic version for testing setup",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
documents = []
search_results = []

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Document Q&A Engine API is running!", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "API server is running"
    }

@app.post("/documents")
async def upload_document_simple(file: UploadFile = File(...)):
    """Simple document upload endpoint"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        content = await file.read()
        
        # Simple document storage
        doc = {
            "id": len(documents) + 1,
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1] if '.' in file.filename else "unknown",
            "file_size": len(content),
            "upload_time": time.time(),
            "processed": True,
            "total_chunks": 1,  # Simplified
            "content_preview": content.decode('utf-8', errors='ignore')[:200] + "..." if len(content) > 200 else content.decode('utf-8', errors='ignore')
        }
        
        documents.append(doc)
        
        return {
            "id": doc["id"],
            "filename": doc["filename"],
            "file_type": doc["file_type"],
            "file_size": doc["file_size"],
            "upload_time": doc["upload_time"],
            "processed": True,
            "total_chunks": 1,
            "processing_time": 0.1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all documents"""
    return documents

@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    """Get specific document"""
    doc = next((d for d in documents if d["id"] == document_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.post("/search")
async def search_documents(request: Dict[str, Any]):
    """Simple search endpoint"""
    query = request.get("query", "")
    max_results = request.get("max_results", 5)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Store search query for analytics
    search_results.append({
        "query": query,
        "timestamp": time.time(),
        "results_count": 0
    })
    
    # Simple keyword search
    results = []
    for doc in documents:
        content = doc.get("content_preview", "")
        if query.lower() in content.lower():
            results.append({
                "id": len(results) + 1,
                "document_id": doc["id"],
                "chunk_index": 0,
                "content": content,
                "token_count": len(content.split()),
                "page_number": 1,
                "relevance_score": 0.9 if query.lower() in content.lower()[:100] else 0.6
            })
    
    # Update search results count
    if search_results:
        search_results[-1]["results_count"] = len(results)
    
    return {
        "query": query,
        "answer": f"Based on your query '{query}', I found {len(results)} relevant passages in the uploaded documents." if results else "I couldn't find relevant information for your query. Try uploading more documents or rephrasing your question.",
        "relevant_chunks": results[:max_results],
        "search_time_ms": 50.0,
        "total_time_ms": 100.0,
        "document_count": len(set(r["document_id"] for r in results))
    }

@app.get("/analytics/stats")
async def get_stats():
    """Get basic statistics"""
    return {
        "search_engine_stats": {
            "search_stats": {"total_searches": len(search_results), "avg_search_time": 75.5},
            "vector_store_stats": {"total_vectors": len(documents), "total_documents": len(documents)},
            "embedding_model": "simple-keyword-search",
            "embedding_dimension": 0
        },
        "query_stats": {
            "total_queries": len(search_results),
            "average_search_time_ms": 75.5
        }
    }

@app.get("/analytics/popular-queries")
async def get_popular_queries(limit: int = 10):
    """Get most popular queries"""
    # Mock popular queries for demo
    popular = [
        {"query": "What is the main topic?", "count": 5},
        {"query": "Summarize key points", "count": 3},
        {"query": "What are the recommendations?", "count": 2},
        {"query": "How does this work?", "count": 1}
    ]
    
    return {"popular_queries": popular[:limit]}

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document"""
    global documents
    doc = next((d for d in documents if d["id"] == document_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    documents = [d for d in documents if d["id"] != document_id]
    return {"message": "Document deleted successfully"}

@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: int, skip: int = 0, limit: int = 100):
    """Get chunks for a document"""
    doc = next((d for d in documents if d["id"] == document_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Mock chunks
    chunks = [{
        "id": 1,
        "chunk_index": 0,
        "content": doc.get("content_preview", ""),
        "token_count": len(doc.get("content_preview", "").split()),
        "page_number": 1,
        "metadata": {"document_id": document_id}
    }]
    
    return {"chunks": chunks, "total": len(chunks)}

@app.post("/admin/clear-cache")
async def clear_cache():
    """Clear application cache"""
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)