# DocQuery AI - Terminal Commands Reference

## 🚀 Start Services

### Start Streamlit Web Interface:
```powershell
cd "c:\project\DocQuery AI"
.venv\Scripts\activate
python -m streamlit run streamlit_app.py --server.port 8503
```

### Start Simple API Server:
```powershell
cd "c:\project\DocQuery AI" 
.venv\Scripts\activate
python simple_api.py
```

### Start Full AI-Powered API:
```powershell
cd "c:\project\DocQuery AI"
.venv\Scripts\activate  
python run_api.py
```

## 🔍 Test the API

### Health Check:
```powershell
curl http://localhost:8000/health
```

### Search Documents:
```powershell
curl "http://localhost:8000/search?query=python"
curl "http://localhost:8000/search?query=machine%20learning"
```

### List Documents:
```powershell
curl http://localhost:8000/documents
```

## 📄 Work with Documents

### View Sample Documents (if available):
```powershell
# Check if sample documents exist
if (Test-Path sample_docs) { dir sample_docs } else { echo "No sample documents - upload your own!" }

# View sample documents (if they exist)
if (Test-Path sample_docs\python_guide.txt) { type sample_docs\python_guide.txt }
if (Test-Path sample_docs\ml_fundamentals.txt) { type sample_docs\ml_fundamentals.txt }
if (Test-Path sample_docs\api_best_practices.txt) { type sample_docs\api_best_practices.txt }
```

### Upload Document via API:
```powershell
# Upload any document you have
curl -X POST "http://localhost:8000/upload" -F "file=@path\to\your\document.pdf"
```

## 🧪 System Testing

### Run Comprehensive Tests:
```powershell
cd "c:\project\DocQuery AI"
.venv\Scripts\activate
python explore_system.py
```

### Check Instruction Compliance:
```powershell
python verify_compliance.py
```

### Quick Status Check:
```powershell
python -c "from app.config import get_settings; s = get_settings(); print(f'Model: {s.embedding_model}')"
```

## 🌐 Access Web Interfaces

- **Streamlit App**: http://localhost:8502 (currently running)
- **API Documentation**: http://localhost:8000/docs (when API is running)
- **Health Check**: http://localhost:8000/health

## 🛠️ Development Commands

### Install New Packages:
```powershell
cd "c:\project\DocQuery AI"
.venv\Scripts\activate
pip install package-name
```

### Check Python Environment:
```powershell
.venv\Scripts\activate
python --version
pip list
```

### View Logs:
```powershell
# API logs are displayed in terminal when running
# Streamlit logs are in browser console
```

## 🎯 Quick Start

1. **Access the working web interface**: http://localhost:8502
2. **Upload documents** through the web interface
3. **Search and test** the Q&A functionality  
4. **For API testing**, start the simple API and use curl commands above

## 💡 Tips

- The Streamlit interface is already running and working
- Use the web interface for easy document upload and search
- Use terminal/API for programmatic access
- Check `verify_compliance.py` output to see all implemented features