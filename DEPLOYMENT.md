# DocQuery AI - Docker Deployment Guide

## 🚀 Quick Start with Docker

### 1. Simple Docker Build & Run
```bash
# Build the Docker image
docker build -t docquery-ai .

# Run the container
docker run -p 8000:8000 -p 8501:8501 docquery-ai
```

### 2. Full Stack with Docker Compose (Recommended)
```bash
# Start everything (app + database + redis)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

## 🌐 Cloud Deployment Options

### Option A: Heroku
```bash
# Install Heroku CLI, then:
heroku create your-docquery-app
heroku stack:set container
git push heroku main
```

### Option B: Railway
```bash
# Connect your GitHub repo to Railway
# Railway will auto-detect Dockerfile and deploy
```

### Option C: Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/docquery-ai
gcloud run deploy --image gcr.io/PROJECT-ID/docquery-ai --platform managed
```

### Option D: AWS ECS/Fargate
```bash
# Use AWS CLI to deploy Docker container
aws ecs create-service --service-name docquery-ai --task-definition docquery-ai
```

## 🔧 Environment Setup

### Required Environment Variables:
```bash
# Optional: For advanced AI features
OPENAI_API_KEY=your_openai_key_here

# Database (if using PostgreSQL)
DATABASE_URL=postgresql://user:pass@host:port/db

# Redis (for caching)
REDIS_URL=redis://host:port
```

## 📱 Access Your Deployed App:
- **Web Interface**: http://your-domain:8501
- **API**: http://your-domain:8000
- **API Docs**: http://your-domain:8000/docs

## 🎯 Production Tips:
1. Set environment variables properly
2. Use a reverse proxy (nginx) for production
3. Enable HTTPS with SSL certificates
4. Set up monitoring and logging
5. Configure backup for your vector store