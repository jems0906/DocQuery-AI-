FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Railway
ENV PYTHONPATH=/app
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with minimal packages first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn requests python-dotenv pydantic loguru && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data safely
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" || echo "NLTK setup completed"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads vector_store

# Create simple startup script
RUN echo '#!/bin/sh\necho "🚀 Starting DocQuery AI on Railway..."\npython railway_start.py' > start.sh && chmod +x start.sh

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["./start.sh"]