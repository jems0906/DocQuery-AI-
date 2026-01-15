#!/bin/bash
# DocQuery AI - Server Deployment Script

echo "🚀 Deploying DocQuery AI to Server"
echo "=================================="

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv nginx

# Clone or copy your project
# cd /var/www/
# git clone your-repo docquery-ai

# Set up virtual environment
cd /var/www/docquery-ai
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your settings

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create systemd services
sudo tee /etc/systemd/system/docquery-api.service > /dev/null <<EOF
[Unit]
Description=DocQuery AI API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/docquery-ai
Environment=PATH=/var/www/docquery-ai/.venv/bin
ExecStart=/var/www/docquery-ai/.venv/bin/python run_api.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/docquery-web.service > /dev/null <<EOF
[Unit]
Description=DocQuery AI Web Interface
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/docquery-ai
Environment=PATH=/var/www/docquery-ai/.venv/bin
ExecStart=/var/www/docquery-ai/.venv/bin/python run_streamlit.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start services
sudo systemctl daemon-reload
sudo systemctl enable docquery-api
sudo systemctl enable docquery-web
sudo systemctl start docquery-api
sudo systemctl start docquery-web

# Configure nginx
sudo tee /etc/nginx/sites-available/docquery-ai > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    # Web Interface
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/docquery-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

echo "✅ Deployment complete!"
echo "🌐 Access your app at: http://your-domain.com"
echo "📡 API available at: http://your-domain.com/api/"