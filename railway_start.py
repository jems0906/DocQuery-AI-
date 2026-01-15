#!/usr/bin/env python3
"""
Railway-optimized startup script for DocQuery AI
"""
import os
import uvicorn
from simple_api import app

if __name__ == "__main__":
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting DocQuery AI on port {port}")
    print("🌐 API will be available at your Railway URL")
    print("📚 API documentation at your Railway URL + /docs")
    
    # Start the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )