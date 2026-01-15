#!/usr/bin/env python3
"""Startup script for the Document Q&A Streamlit web interface."""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from app.config import get_settings
    
    settings = get_settings()
    
    print(f"Starting Document Q&A Engine Web Interface...")
    print(f"Web interface will be available at: http://localhost:{settings.web_port}")
    print(f"Make sure the API server is running at: http://{settings.api_host}:{settings.api_port}")
    
    # Run Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app.py",
        "--server.port", str(settings.web_port),
        "--server.address", "localhost",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ])