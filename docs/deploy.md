# Deployment Guide - SC Planner

Production deployment guide for Streamlit apps with Supabase authentication.

---

## Production Deployment Options

### Option 1: Streamlit Community Cloud (Recommended for Testing)
**Best for**: Quick demos, testing, small teams  
**Cost**: Free  
**Limitations**: Public apps only, limited resources

### Option 2: AWS EC2 / DigitalOcean / Linode (Recommended for Production)
**Best for**: Production apps, full control, custom domains  
**Cost**: $5-20/month  
**Limitations**: Requires server management

### Option 3: Docker + Cloud Run / ECS
**Best for**: Scalable production, containerized deployments  
**Cost**: Pay per use  
**Limitations**: More complex setup

### Option 4: Railway / Render
**Best for**: Easy production deployment, managed hosting  
**Cost**: $5-20/month  
**Limitations**: Less control than VPS

---

## Quick Start: Streamlit Community Cloud

**Use for testing only. Not recommended for production.**

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → Select repo
4. Main file: `run_app.py`
5. Add secrets:
```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
SUPABASE_ANON_KEY = "your-anon-key"
SUPABASE_SCHEMA = "eurofred"
AUTH_TABLE = "auth"
```
6. Deploy

**Note**: The app will automatically prompt you to create the auth table on first run.

---

## Production Deployment: VPS (Recommended)

### Prerequisites
- VPS (DigitalOcean, AWS EC2, Linode, etc.)
- Ubuntu 22.04 LTS
- Domain name (optional)
- Supabase account

### Step 1: Setup Supabase

1. Create project at [supabase.com](https://supabase.com)
2. Get credentials from Settings → API:
   - Project URL
   - anon/public key
3. Keep these for later

### Step 2: Prepare Server

```bash
# SSH into your server
ssh root@your-server-ip

# Update system
apt update && apt upgrade -y

# Install Python 3.10+
apt install python3.10 python3-pip python3-venv -y

# Install nginx (for reverse proxy)
apt install nginx -y

# Install certbot (for SSL)
apt install certbot python3-certbot-nginx -y
```

### Step 3: Deploy Application

```bash
# Create app user
adduser streamlit
usermod -aG sudo streamlit
su - streamlit

# Clone repository
git clone https://github.com/yourusername/sc-planner.git
cd sc-planner

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
nano .env
```

Add to `.env`:
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SCHEMA=eurofred
AUTH_TABLE=auth
PYTHONDONTWRITEBYTECODE=1
```

```bash
# Test app (will prompt to create auth table on first run)
streamlit run run_app.py --server.port 8501
# Visit http://your-server-ip:8501 to test
# Follow prompts to create auth table
# Ctrl+C to stop
```

### Step 4: Setup Systemd Service

```bash
# Exit to root
exit

# Create service file
nano /etc/systemd/system/streamlit.service
```

Add:
```ini
[Unit]
Description=Streamlit SC Planner
After=network.target

[Service]
Type=simple
User=streamlit
WorkingDirectory=/home/streamlit/sc-planner
Environment="PATH=/home/streamlit/sc-planner/venv/bin"
ExecStart=/home/streamlit/sc-planner/venv/bin/streamlit run run_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
systemctl daemon-reload
systemctl enable streamlit
systemctl start streamlit
systemctl status streamlit
```

### Step 5: Setup Nginx Reverse Proxy

```bash
nano /etc/nginx/sites-available/streamlit
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # or your-server-ip

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

```bash
# Enable site
ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

### Step 6: Setup SSL (Optional but Recommended)

```bash
# Get SSL certificate
certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
certbot renew --dry-run
```

### Step 7: Verify Deployment

Visit `https://your-domain.com` (or `http://your-server-ip`)

Test:
- Login with admin/admin123
- Create new user via signup
- Test all features
- Check logs: `journalctl -u streamlit -f`

---

## Alternative: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "run_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
      - SUPABASE_SCHEMA=eurofred
      - AUTH_TABLE=auth
    volumes:
      - ./output:/app/output
    restart: unless-stopped
```

### Deploy

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Railway Deployment

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login and deploy:
```bash
railway login
railway init
railway up
```

3. Add environment variables:
```bash
railway variables set SUPABASE_URL=https://your-project.supabase.co
railway variables set SUPABASE_KEY=your-anon-key
railway variables set SUPABASE_ANON_KEY=your-anon-key
railway variables set SUPABASE_SCHEMA=eurofred
railway variables set AUTH_TABLE=auth
```

4. Create `railway.json`:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run run_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

---

## Environment Variables

Required for all deployments:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SCHEMA=eurofred
AUTH_TABLE=auth
```

Optional:
```bash
PYTHONDONTWRITEBYTECODE=1
```

---

## Initial Setup

**After first deployment:**

`run_app.py` automatically checks if the auth table exists and offers to create it.

**Option 1: Automatic (recommended)**
```bash
# Just run - it will prompt you to create the table if needed
python run_app.py
# or
streamlit run run_app.py
```

**Option 2: Manual**
```bash
# Create table manually first
python src/setup_auth.py

# Then run app
python run_app.py
```

This creates:
- `eurofred.auth` table in Supabase
- Admin user: admin/admin123
- Demo user: demo/demo123

**⚠️ Change default passwords immediately!**

---

## Monitoring & Maintenance

### Check Logs

**Systemd:**
```bash
journalctl -u streamlit -f
```

**Docker:**
```bash
docker-compose logs -f
```

**Railway:**
```bash
railway logs
```

### Update Application

```bash
# Pull latest code
cd /home/streamlit/sc-planner
git pull

# Restart service
sudo systemctl restart streamlit
```

### Backup

**Supabase** (automatic backups on paid plans)

**Application data:**
```bash
# Backup output directory
tar -czf backup-$(date +%Y%m%d).tar.gz output/
```

---

## Security Checklist

- [ ] SSL certificate installed (HTTPS)
- [ ] Firewall configured (ufw/iptables)
- [ ] SSH key authentication only
- [ ] Regular system updates
- [ ] Strong passwords for default users
- [ ] `.env` file permissions: `chmod 600 .env`
- [ ] Supabase RLS policies configured
- [ ] Regular backups scheduled

---

## Troubleshooting

### App won't start
```bash
# Check logs
journalctl -u streamlit -n 50

# Check if port is in use
netstat -tulpn | grep 8501

# Test manually
cd /home/streamlit/sc-planner
source venv/bin/activate
streamlit run run_app.py
```

### Can't connect to Supabase
- Verify credentials in `.env`
- Check Supabase project is active
- Test connection: `curl https://your-project.supabase.co`

### Auth table doesn't exist
```bash
python src/setup_auth.py
```

---

## Cost Estimates

**VPS (DigitalOcean/Linode):**
- Basic: $6/month (1GB RAM)
- Standard: $12/month (2GB RAM)
- Performance: $24/month (4GB RAM)

**Railway:**
- Free: $5 credit
- Hobby: $5/month
- Pro: $20/month

**Supabase:**
- Free: 50K MAU (sufficient for auth)
- Pro: $25/month (100K+ users)

**Recommended for production:** VPS ($12/month) + Supabase Free = $12/month total

---

## Support

- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io)
- **Supabase**: [supabase.com/docs](https://supabase.com/docs)
- **DigitalOcean**: [docs.digitalocean.com](https://docs.digitalocean.com)

---

**Last Updated**: November 12, 2025
