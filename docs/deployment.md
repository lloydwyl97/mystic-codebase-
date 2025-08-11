# Mystic Trading Platform - Deployment Guide

## Overview

This guide covers deploying the Mystic Trading Platform with both backend (FastAPI) and frontend (React) components.

## Architecture

- **Backend**: FastAPI on port 8000
- **Frontend**: React (served by nginx) on port 80
- **Database**: SQLite (can be upgraded to PostgreSQL)
- **Cache**: Redis on port 6379
- **WebSocket**: Real-time data streaming

## Quick Start with Docker Compose

### Prerequisites

- Docker and Docker Compose installed
- At least 2GB RAM available
- Ports 80, 8000, and 6379 available

### 1. Clone and Setup

```bash
git clone <your-repo>
cd Mystic-Codebase
```

### 2. Environment Configuration

```bash
# Copy and edit environment file
cp backend/env.example backend/.env
# Edit backend/.env with your production settings
```

### 3. Deploy with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Access the Application

- **Frontend**: <http://localhost>
- **Backend API**: <http://localhost:8000>
- **Health Check**: <http://localhost:8000/health>

## Manual Deployment

### Backend Deployment

#### 1. Install Backend Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Configure Environment

```bash
cp env.example .env
# Edit .env with your settings
```

#### 3. Start Backend

```bash
# Development
python main.py

# Production with Gunicorn
gunicorn --config gunicorn.conf.py main:app
```

### Frontend Deployment

#### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

#### 2. Build for Production

```bash
npm run build
```

#### 3. Serve with nginx

```bash
# Copy built files to nginx directory
sudo cp -r dist/* /var/www/html/

# Use the provided nginx.conf
sudo cp nginx.conf /etc/nginx/nginx.conf
sudo systemctl restart nginx
```

## Production Configuration

### Environment Variables

#### Backend (.env)

```bash
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your-super-secret-key
REDIS_URL=redis://localhost:6379
DATABASE_URL=sqlite:///./mystic_trading.db
LOG_LEVEL=INFO
USE_MOCK_DATA=false
TRADING_ENABLED=false
```

#### Frontend (.env)

```bash
VITE_API_URL=http://your-domain.com:8000
```

### Security Considerations

1. **Change Default Secrets**
   - Update `SECRET_KEY` in backend
   - Use strong, unique keys

2. **Enable HTTPS**
   - Configure SSL certificates
   - Update CORS origins to HTTPS

3. **Database Security**
   - Use PostgreSQL in production
   - Implement proper backup strategy

4. **API Security**
   - Implement rate limiting
   - Add authentication middleware
   - Use API keys for external access

### Monitoring and Logging

#### Backend Logs

```bash
# View application logs
tail -f backend/logs/mystic_trading.log

# View Gunicorn logs
docker-compose logs backend
```

#### Frontend Logs

```bash
# View nginx logs
docker-compose logs frontend
# or
sudo tail -f /var/log/nginx/access.log
```

### Scaling Considerations

#### Horizontal Scaling

```bash
# Scale backend workers
docker-compose up -d --scale backend=3

# Use load balancer for multiple instances
```

#### Database Scaling

- Migrate from SQLite to PostgreSQL
- Implement database clustering
- Add read replicas for analytics

## Troubleshooting

### Common Issues

#### 1. Port Conflicts

```bash
# Check what's using the ports
netstat -tulpn | grep :8000
netstat -tulpn | grep :80

# Kill conflicting processes
sudo kill -9 <PID>
```

#### 2. Permission Issues

```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x scripts/*.sh
```

#### 3. Database Issues

```bash
# Reset database
rm backend/mystic_trading.db
docker-compose restart backend
```

#### 4. Build Issues

```bash
# Clean and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Health Checks

#### Backend Health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/health/comprehensive
```

#### Frontend Health

```bash
curl http://localhost/
```

#### Redis Health

```bash
docker-compose exec redis redis-cli ping
```

## Performance Optimization

### Backend

- Enable connection pooling
- Implement caching strategies
- Optimize database queries
- Use async operations

### Frontend

- Enable gzip compression
- Implement lazy loading
- Use CDN for static assets
- Optimize bundle size

## Backup and Recovery

### Database Backup

```bash
# SQLite backup
cp backend/mystic_trading.db backend/mystic_trading.db.backup

# Redis backup
docker-compose exec redis redis-cli BGSAVE
```

### Application Backup

```bash
# Create backup archive
tar -czf mystic-backup-$(date +%Y%m%d).tar.gz \
  backend/mystic_trading.db \
  backend/logs/ \
  frontend/dist/
```

## Support

For issues and questions:

1. Check the logs first
2. Review this deployment guide
3. Check the application documentation
4. Create an issue in the repository
