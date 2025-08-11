# Mystic Trading Platform - Caddy Deployment Guide

## 🚀 Quick Start with Caddy

Caddy is a modern web server with automatic HTTPS, making it perfect for deploying the Mystic Trading Platform.

## 📋 Prerequisites

- Linux/macOS system
- Python 3.8+
- Node.js 16+
- sudo privileges
- Domain name (for production)

## 🛠️ Installation Steps

### 1. Install Caddy

```bash
# Run the installation script
chmod +x install-caddy.sh
./install-caddy.sh
```

Or install manually:

**Ubuntu/Debian:**

```bash
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

**macOS:**

```bash
brew install caddy
```

### 2. Deploy the Application

```bash
# Run the deployment script
chmod +x deploy-caddy.sh
./deploy-caddy.sh
```

This script will:

- ✅ Build the frontend
- ✅ Configure Caddy
- ✅ Set up systemd services
- ✅ Start both frontend and backend
- ✅ Run health checks

## 🌐 Access Your Application

After deployment:

- **Frontend**: <http://localhost>
- **Backend API**: <http://localhost:8000>
- **Health Check**: <http://localhost/health>
- **API Documentation**: <http://localhost:8000/docs>

## 🔧 Configuration Files

### Development (localhost)

- **File**: `Caddyfile`
- **Purpose**: Local development setup

### Production (domain)

- **File**: `Caddyfile.production`
- **Purpose**: Production deployment with domain

## 🚀 Production Deployment

### 1. Update Domain Configuration

Edit `Caddyfile.production`:

```caddyfile
# Replace 'yourdomain.com' with your actual domain
yourdomain.com {
    # ... rest of configuration
}
```

### 2. Deploy to Production

```bash
# Copy production configuration
sudo cp Caddyfile.production /etc/caddy/Caddyfile

# Reload Caddy
sudo systemctl reload caddy
```

### 3. SSL Certificates

Caddy automatically:

- ✅ Obtains SSL certificates from Let's Encrypt
- ✅ Renews certificates automatically
- ✅ Redirects HTTP to HTTPS
- ✅ Handles certificate management

## 📊 Monitoring and Logs

### View Logs

```bash
# Caddy logs
sudo journalctl -u caddy -f

# Backend logs
sudo journalctl -u mystic-backend -f

# Application logs
tail -f /var/log/caddy/mystic.log
```

### Health Checks

```bash
# Frontend health
curl -f http://localhost/

# Backend health
curl -f http://localhost:8000/health

# API proxy health
curl -f http://localhost/api/health
```

## 🔧 Management Commands

### Service Management

```bash
# Start services
sudo systemctl start caddy mystic-backend

# Stop services
sudo systemctl stop caddy mystic-backend

# Restart services
sudo systemctl restart caddy mystic-backend

# Enable auto-start
sudo systemctl enable caddy mystic-backend

# Check status
sudo systemctl status caddy mystic-backend
```

### Caddy Management

```bash
# Test configuration
sudo caddy validate --config /etc/caddy/Caddyfile

# Reload configuration
sudo systemctl reload caddy

# View Caddy admin API (if enabled)
curl http://localhost:2019/config/
```

## 🔒 Security Features

### Automatic Security

Caddy provides:

- ✅ **Automatic HTTPS** with Let's Encrypt
- ✅ **HTTP/2** and **HTTP/3** support
- ✅ **Security headers** (HSTS, CSP, etc.)
- ✅ **Rate limiting** for API endpoints
- ✅ **WebSocket security**

### Manual Security Enhancements

1. **Update CORS origins** in backend config
2. **Change default secrets** in `.env`
3. **Configure firewall** rules
4. **Set up monitoring** and alerts

## 📈 Performance Optimization

### Caddy Optimizations

- ✅ **Gzip compression** enabled
- ✅ **Static file caching** configured
- ✅ **Connection pooling** for backend
- ✅ **Load balancing** ready (commented)

### Backend Optimizations

- ✅ **Multiple workers** with Gunicorn
- ✅ **Connection pooling** for database
- ✅ **Caching** with Redis
- ✅ **Async operations** enabled

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Conflicts

```bash
# Check what's using ports
sudo netstat -tulpn | grep :80
sudo netstat -tulpn | grep :443
sudo netstat -tulpn | grep :8000

# Kill conflicting processes
sudo kill -9 <PID>
```

#### 2. Permission Issues

```bash
# Fix file permissions
sudo chown -R caddy:caddy /var/log/caddy
sudo chown -R caddy:caddy /etc/caddy
sudo chown -R $USER:$USER /var/www/mystic
```

#### 3. SSL Certificate Issues

```bash
# Check certificate status
sudo caddy certificates

# Force certificate renewal
sudo systemctl reload caddy
```

#### 4. Backend Connection Issues

```bash
# Check backend logs
sudo journalctl -u mystic-backend --no-pager -n 50

# Test backend directly
curl http://localhost:8000/health

# Check environment
cd backend
source venv/bin/activate
python -c "import main; print('Backend imports OK')"
```

### Debug Mode

Enable debug logging:

```bash
# Edit Caddyfile
sudo nano /etc/caddy/Caddyfile

# Add debug logging
log {
    output stdout
    level DEBUG
}

# Reload Caddy
sudo systemctl reload caddy
```

## 🔄 Updates and Maintenance

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild frontend
cd frontend
npm install
npm run build
sudo cp -r dist/* /var/www/mystic/

# Restart backend
sudo systemctl restart mystic-backend

# Reload Caddy
sudo systemctl reload caddy
```

### Update Caddy

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade caddy

# macOS
brew upgrade caddy

# Restart Caddy
sudo systemctl restart caddy
```

## 📚 Additional Resources

- [Caddy Documentation](https://caddyserver.com/docs/)
- [Caddy Configuration Examples](https://github.com/caddyserver/examples)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)

## 🆘 Support

For issues:

1. Check the logs first
2. Review this deployment guide
3. Check Caddy documentation
4. Create an issue in the repository

---

## Happy Trading! 📈
