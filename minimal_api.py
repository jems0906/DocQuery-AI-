#!/usr/bin/env python3
"""
Super simple DocQuery AI API for Railway deployment
"""
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocQuery AI",
    description="Document Q&A Engine with embeddings",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "🚀 DocQuery AI is running!",
        "status": "online",
        "version": "1.0.0",
        "description": "Document Q&A Engine with embeddings"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DocQuery AI",
        "version": "1.0.0"
    }

@app.get("/docs-info")
async def docs_info():
    """Information about the API documentation"""
    return {
        "message": "API Documentation available at /docs",
        "endpoints": [
            "GET / - Welcome message",
            "GET /health - Health check",
            "GET /docs - API documentation"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting DocQuery AI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")