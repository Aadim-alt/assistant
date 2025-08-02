# JARVIS AI Assistant - Production Deployment Guide

## 📋 Table of Contents
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Security Setup](#security-setup)
- [Deployment Options](#deployment-options)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+
- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 20 GB free space
- **Python**: 3.9+
- **GPU**: Optional (CUDA-compatible for AI acceleration)

### Recommended Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 32 GB
- **Storage**: 50 GB SSD
- **GPU**: NVIDIA RTX series with 8GB+ VRAM
- **Network**: Stable internet connection

## 🚀 Installation

### 1. Quick Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-org/jarvis-ai.git
cd jarvis-ai

# Run automated setup
python setup.py --mode production

# Install system dependencies
./scripts/install_system_deps.sh

# Install Python dependencies
pip install -r requirements.txt

# Download AI models
python -m jarvis.setup download-models
```

### 2. Manual Installation

#### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv jarvis-env
source jarvis-env/bin/activate  # Linux/Mac
# jarvis-env\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 2: Core Dependencies
```bash
# AI/ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
pip install ollama

# Voice processing
pip install faster-whisper openai-whisper
pip install pyttsx3 SpeechRecognition pyaudio

# Computer vision
pip install opencv-python pillow pytesseract
pip install ultralytics  # YOLO for object detection

# GUI and web
pip install customtkinter
pip install fastapi uvicorn websockets

# System utilities
pip install psutil GPUtil py-cpuinfo
pip install watchdog schedule apscheduler

# Security
pip install cryptography keyring
pip install aiohttp aiofiles

# Additional utilities
pip install python-dotenv configparser
pip install pytest pytest-asyncio pytest-cov  # For testing
```

#### Step 3: System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y tesseract-ocr espeak portaudio19-dev
sudo apt install -y python3-tk python3-dev build-essential

# macOS (with Homebrew)
brew install tesseract espeak portaudio
xcode-select --install

# Windows
# Download and install:
# - Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
# - Microsoft Visual C++ Build Tools
```

### 3. Model Installation

#### Ollama Models (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download models
ollama pull llama2
ollama pull mistral
ollama pull codellama

# Verify installation
ollama list
```

#### Whisper Models
```bash
# Download Whisper models
python -c "import whisper; whisper.load_model('base')"
python -c "import whisper; whisper.load_model('small')"
```

#### spaCy Models
```bash
# Download language models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md  # Optional, larger model
```

## ⚙️ Configuration

### 1. Environment Variables
Create `.env` file in project root:
```bash
# Core Configuration
JARVIS_MASTER_NAME="Your Name"
JARVIS_AI_NAME="JARVIS"
JARVIS_WAKE_WORD="jarvis"

# AI Models
JARVIS_LLM_MODEL="llama2"
JARVIS_VOICE_MODEL="whisper-base"

# API Keys (Optional but recommended)
JARVIS_OPENWEATHER_API_KEY="your_openweather_key"
JARVIS_NEWSAPI_API_KEY="your_news_api_key"
JARVIS_WOLFRAM_API_KEY="your_wolfram_key"

# Security
JARVIS_SECURE_MODE="true"
JARVIS_ENCRYPTION_ENABLED="true"

# Performance
JARVIS_MAX_WORKERS="4"
JARVIS_CACHE_SIZE="1000"
JARVIS_LOG_LEVEL="INFO"

# Web API
JARVIS_WEB_API_HOST="127.0.0.1"
JARVIS_WEB_API_PORT="8000"
JARVIS_API_RATE_LIMIT="100"  # requests per minute

# Resource Limits
JARVIS_MAX_MEMORY_MB="2048"
JARVIS_MAX_CPU_PERCENT="80"
```

### 2. JARVIS Configuration
Create `config/jarvis.json`:
```json
{
  "master_name": "Your Name",
  "ai_name": "JARVIS",
  "wake_word": "jarvis",
  "voice_model": "whisper-base",
  "llm_model": "llama2",
  "voice_rate": 180,
  "voice_volume": 0.8,
  "language": "en",
  "theme": "dark",
  "api_timeout": 30,
  "max_context_length": 4096,
  "enable_vision": true,
  "enable_automation": true,
  "enable_learning": true,
  "debug_mode": false,
  "features": {
    "voice_wake_word": true,
    "gui_interface": true,
    "web_api": true,
    "plugin_system": true,
    "system_monitoring": true,
    "automation_engine": true,
    "computer_vision": true
  },
  "performance": {
    "lazy_loading": true,
    "model_caching": true,
    "response_caching": true,
    "background_tasks": true
  }
}
```

### 3. Logging Configuration
Create `config/logging.json`:
```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "standard": {
      "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    },
    "detailed": {
      "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "INFO",
      "formatter": "standard",
      "stream": "ext://sys.stdout"
    },
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "DEBUG",
      "formatter": "detailed",
      "filename": "logs/jarvis.log",
      "maxBytes": 10485760,
      "backupCount": 5
    },
    "error_file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "ERROR",
      "formatter": "detailed",
      "filename": "logs/jarvis_errors.log",
      "maxBytes": 10485760,
      "backupCount": 3
    }
  },
  "loggers": {
    "jarvis": {
      "level": "DEBUG",
      "handlers": ["console", "file", "error_file"],
      "propagate": false
    }
  },
  "root": {
    "level": "INFO",
    "handlers": ["console"]
  }
}
```

## 🔒 Security Setup

### 1. API Key Management
```bash
# Store API keys securely
python -c "
from jarvis.core.security import SecurityManager
sm = SecurityManager()
sm.store_api_key('openweather', 'your_api_key_here')
sm.store_api_key('newsapi', 'your_news_api_key_here')
"
```

### 2. SSL/TLS for Web API
```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# For production, use Let's Encrypt or commercial certificate
certbot certonly --standalone -d your-domain.com
```

### 3. Firewall Configuration
```bash
# Ubuntu/Debian
sudo ufw allow 8000/tcp  # JARVIS Web API
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### 4. User Permissions
```bash
# Create dedicated user for JARVIS
sudo useradd -r -s /bin/false jarvis
sudo mkdir -p /opt/jarvis
sudo chown jarvis:jarvis /opt/jarvis

# Set up proper permissions
chmod 750 /opt/jarvis
chmod 600 /opt/jarvis/.env
```

## 🚀 Deployment Options

### Option 1: Systemd Service (Linux)

Create `/etc/systemd/system/jarvis.service`:
```ini
[Unit]
Description=JARVIS AI Assistant
After=network.target
Wants=network.target

[Service]
Type=simple
User=jarvis
Group=jarvis
WorkingDirectory=/opt/jarvis
Environment=PATH=/opt/jarvis/jarvis-env/bin
ExecStart=/opt/jarvis/jarvis-env/bin/python main.py --mode api
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=jarvis

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/jarvis

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable jarvis
sudo systemctl start jarvis
sudo systemctl status jarvis
```

### Option 2: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    espeak \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -r -s /bin/false jarvis && \
    chown -R jarvis:jarvis /app

USER jarvis

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "main.py", "--mode", "api", "--host", "0.0.0.0"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  jarvis:
    build: .
    ports:
      - "8000:8000"
    environment:
      - JARVIS_LOG_LEVEL=INFO
      - JARVIS_WEB_API_HOST=0.0.0.0
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs:ro
    depends_on:
      - jarvis
    restart: unless-stopped
```

Deploy:
```bash
docker-compose up -d
docker-compose logs -f jarvis
```

### Option 3: Process Manager (PM2)

Install PM2:
```bash
npm install -g pm2
```

Create `ecosystem.config.js`:
```javascript
module.exports = {
  apps: [{
    name: 'jarvis-api',
    script: 'main.py',
    args: '--mode api',
    interpreter: 'python3',
    cwd: '/opt/jarvis',
    instances: 1,
    exec_mode: 'fork',
    watch: false,
    max_memory_restart: '2G',
    env: {
      NODE_ENV: 'production',
      JARVIS_LOG_LEVEL: 'INFO'
    },
    log_file: '/opt/jarvis/logs/pm2.log',
    error_file: '/opt/jarvis/logs/pm2_error.log',
    out_file: '/opt/jarvis/logs/pm2_out.log',
    time: true
  }]
};
```

Start with PM2:
```bash
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

## 📊 Monitoring & Maintenance

### 1. Health Monitoring Script
Create `scripts/health_check.py`:
```python
#!/usr/bin/env python3

import asyncio
import aiohttp
import logging
import sys
from datetime import datetime

async def check_jarvis_health():
    """Check JARVIS health endpoints"""
    endpoints = [
        'http://localhost:8000/health',
        'http://localhost:8000/api/status',
        'http://localhost:8000/api/system-info'
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                async with session.get(endpoint, timeout=10) as response:
                    if response.status == 200:
                        print(f"✅ {endpoint} - OK")
                    else:
                        print(f"❌ {endpoint} - HTTP {response.status}")
                        return False
            except Exception as e:
                print(f"❌ {endpoint} - Error: {e}")
                return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(check_jarvis_health())
    sys.exit(0 if result else 1)
```

### 2. Log Rotation Setup
```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/jarvis << EOF
/opt/jarvis/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 jarvis jarvis
    postrotate
        systemctl reload jarvis
    endscript
}
EOF
```

### 3. Backup Script
Create `scripts/backup.sh`:
```bash
#!/bin/bash

BACKUP_DIR="/opt/backups/jarvis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="jarvis_backup_${TIMESTAMP}.tar.gz"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create backup
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    --exclude='logs/*' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    /opt/jarvis/

# Keep only last 7 backups
find "$BACKUP_DIR" -name "jarvis_backup_*.tar.gz" -mtime +7 -delete

echo "Backup created: $BACKUP_DIR/$BACKUP_FILE"
```

### 4. Performance Monitoring
Create monitoring with Prometheus/Grafana:

`monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'jarvis'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 5. Automated Updates
Create `scripts/update.sh`:
```bash
#!/bin/bash

set -e

echo "Starting JARVIS update..."

# Stop service
sudo systemctl stop jarvis

# Backup current version
./scripts/backup.sh

# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migrations/updates
python -m jarvis.setup migrate

# Restart service
sudo systemctl start jarvis

# Verify health
sleep 10
python scripts/health_check.py

echo "Update completed successfully!"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```bash
# Check Ollama status
ollama list
systemctl status ollama

# Reinstall models
ollama pull llama2
```

#### 2. Voice Recognition Issues
```bash
# Check audio devices
python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"

# Test microphone
arecord -l  # Linux
```

#### 3. Permission Errors
```bash
# Fix permissions
sudo chown -R jarvis:jarvis /opt/jarvis
chmod +x /opt/jarvis/main.py
```

#### 4. Memory Issues
```bash
# Monitor memory usage
htop
free -h

# Adjust memory limits in config
```

#### 5. API Connection Issues
```bash
# Check port availability
netstat -tuln | grep 8000

# Test API endpoint
curl http://localhost:8000/health
```

### Debug Mode
```bash
# Run in debug mode
python main.py --mode gui --debug

# Check logs
tail -f logs/jarvis.log
journalctl -u jarvis -f
```

### Recovery Procedures

#### 1. Service Recovery
```bash
# If service fails to start
sudo systemctl status jarvis
sudo journalctl -u jarvis --no-pager -l

# Reset configuration
cp config/jarvis.json.backup config/jarvis.json
sudo systemctl restart jarvis
```

#### 2. Database Recovery
```bash
# If using database
python -m jarvis.setup reset-db
python -m jarvis.setup init-db
```

#### 3. Complete Reset
```bash
# Nuclear option - complete reset
sudo systemctl stop jarvis
rm -rf /opt/jarvis/.jarvis/cache
cp config/jarvis.json.default config/jarvis.json
sudo systemctl start jarvis
```

## 📈 Performance Optimization

### 1. Resource Tuning
```bash
# Adjust worker processes
export JARVIS_MAX_WORKERS=8

# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0
```

### 2. Caching Configuration
```python
# In config/performance.json
{
  "cache": {
    "response_cache_size": 1000,
    "response_cache_ttl": 300,
    "model_cache_size": 5,
    "enable_disk_cache": true
  }
}
```

### 3. Database Optimization
```sql
-- If using PostgreSQL
CREATE INDEX idx_conversations_timestamp ON conversations(created_at);
CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);
```

## 🚀 Scaling Considerations

### Horizontal Scaling
- Load balancer (Nginx/HAProxy)
- Multiple JARVIS instances
- Shared cache (Redis)
- Message queue (RabbitMQ/Kafka)

### Vertical Scaling
- Increase memory allocation
- Add GPU acceleration
- SSD storage for models
- Network optimization

---

## 📞 Support

For production support:
- **Documentation**: https://docs.jarvis-ai.com
- **Issues**: https://github.com/Aadil-alt/jarvis-ai/issues
- **Community**: https://discord.gg/jarvis-ai
- **Enterprise Support**: support@jarvis-ai.com

---

**Last Updated**: December 2024
**Version**: 2.0.0