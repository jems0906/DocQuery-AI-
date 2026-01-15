# Railway Deployment for DocQuery AI

## 🚂 Deploy to Railway (Recommended)

1. **Visit**: https://railway.app
2. **Sign up/Login** with your GitHub account
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose**: `jems0906/DocQuery-AI-`
6. **Railway will automatically**:
   - Detect your Dockerfile
   - Build and deploy your app
   - Provide you with a public URL

## ⚙️ Environment Variables (Optional)
In Railway dashboard, add these if you want advanced AI features:
- `OPENAI_API_KEY`: Your OpenAI key for advanced embeddings

## 🎯 After Deployment:
- **Web Interface**: Your Railway URL (e.g., https://docquery-ai-production.up.railway.app)
- **API Docs**: Your Railway URL + `/docs`

## 🔄 Auto-Deploy
Every time you push to GitHub, Railway will automatically redeploy!

```bash
# Make changes locally
git add .
git commit -m "Update feature"
git push origin main
# Railway automatically deploys!
```