FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Railway
ENV PYTHONPATH=/app
ENV PORT=8000

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads vector_store

# Expose ports
EXPOSE 8000 8501

# Create startup script for Railway
RUN echo '#!/bin/bash\n\
echo "Starting DocQuery AI..."\n\
python railway_start.py' > start.sh && chmod +x start.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port Railway expects
EXPOSE $PORT

# Simple healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["./start.sh"]