#!/usr/bin/env python3
"""Simplified startup script for the Document Q&A API server."""

import uvicorn
from pathlib import Path

if __name__ == "__main__":
    # Basic FastAPI app without complex dependencies first
    print("Starting Document Q&A Engine API server...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    
    try:
        uvicorn.run(
            "simple_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("Trying basic API...")
        # Import inline to catch issues
        from simple_api import app
        uvicorn.run(app, host="0.0.0.0", port=8000)