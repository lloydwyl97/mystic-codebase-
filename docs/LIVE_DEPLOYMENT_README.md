# 🚀 Mystic Trading Platform - Live Deployment Guide

## 🌐 Windows 11 Home + Docker Deployment

This guide will help you deploy the complete Mystic Trading Platform live with all endpoints running on Windows 11 Home using Docker.

## 📋 Prerequisites

### Required Software
- **Windows 11 Home** (or Windows 10)
- **Docker Desktop for Windows** (latest version)
- **PowerShell** (built-in)
- **Git** (optional, for updates)

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 10GB free space
- **CPU**: Multi-core processor
- **Network**: Stable internet connection

## 🚀 Quick Start

### 1. Prepare Your Environment

```powershell
# Run as Administrator (recommended)
# Right-click PowerShell and select "Run as Administrator"

# Navigate to your Mystic-Codebase directory
cd C:\path\to\Mystic-Codebase
```

### 2. Set Up Environment Variables

```powershell
# Run the environment setup script
.\setup-env.ps1
```

This will:
- Create `backend/.env` with all necessary configuration
- Set up required directories
- Configure production settings

### 3. Configure API Keys (Optional but Recommended)

Edit `backend/.env` and add your actual API keys:

```env
# Exchange API Keys (for live trading)
BINANCE_API_KEY=your_actual_binance_api_key
BINANCE_SECRET_KEY=your_actual_binance_secret_key
BINANCE_TESTNET=false

COINBASE_API_KEY=your_actual_coinbase_api_key
COINBASE_SECRET_KEY=your_actual_coinbase_secret_key
COINBASE_PASSPHRASE=your_actual_coinbase_passphrase
COINBASE_SANDBOX=false

# AI Services (for enhanced features)
OPENAI_API_KEY=your_actual_openai_api_key

# Notifications (optional)
DISCORD_WEBHOOK=your_discord_webhook_url
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 4. Launch the Platform

```powershell
# Deploy everything live
.\launch-live.ps1
```

## 🎯 What Gets Deployed

### 🌐 Frontend (Port 80)
- **Main Dashboard**: http://localhost
- **Analytics Dashboard**: http://localhost/analytics
- **Trading Dashboard**: http://localhost/trading
- **AI Dashboard**: http://localhost/ai
- **Social Trading**: http://localhost/social

### 🔧 Backend API (Port 8000)
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

#### Trading Endpoints
- Market Data: `http://localhost:8000/api/v1/market/*`
- Trading Status: `http://localhost:8000/api/v1/trading/*`
- AI Decisions: `http://localhost:8000/api/v1/ai/*`
- Bot Management: `http://localhost:8000/api/v1/bots/*`

#### Analytics Endpoints
- Analytics Overview: `http://localhost:8000/api/v1/analytics/*`
- Performance Metrics: `http://localhost:8000/api/v1/analytics/performance`
- Portfolio Analysis: `http://localhost:8000/api/v1/analytics/portfolio`

#### AI & Automation
- AI Trading: `http://localhost:8000/api/v1/ai/trading`
- Strategy Management: `http://localhost:8000/api/v1/strategies`
- Signal Management: `http://localhost:8000/api/v1/signals`

#### Notifications & Social
- Notifications: `http://localhost:8000/api/v1/notifications`
- Social Trading: `http://localhost:8000/api/v1/social`
- WebSocket: `ws://localhost:8000/ws`

### 🔴 Redis (Port 6379)
- Caching and session management
- Real-time data storage
- WebSocket connections

## 🛠️ Management Commands

### View Logs
```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f redis
```

### Container Management
```powershell
# View running containers
docker-compose ps

# Stop all services
docker-compose down

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Access Container Shell
```powershell
# Backend container
docker exec -it mystic-backend bash

# Frontend container
docker exec -it mystic-frontend sh
```

## 🔧 Configuration Options

### Environment Variables

Key configuration options in `backend/.env`:

```env
# Trading Configuration
TRADING_ENABLED=true              # Enable live trading
MAX_POSITION_SIZE=500.0           # Maximum position size
MAX_DAILY_LOSS=2.0               # Maximum daily loss percentage
MAX_CONCURRENT_POSITIONS=3        # Maximum concurrent positions

# AI Configuration
AI_CONFIDENCE_THRESHOLD=0.75      # AI confidence threshold
AI_AUTO_TRAINING=true             # Enable auto-training
AI_TRAINING_INTERVAL=24           # Training interval in hours

# Performance
CACHE_TTL=300                     # Cache time-to-live
RATE_LIMIT_REQUESTS=100           # Rate limit requests per window
RATE_LIMIT_WINDOW=60              # Rate limit window in seconds
```

### Docker Configuration

The platform uses Docker Compose with the following services:

- **redis**: Redis database for caching
- **backend**: FastAPI backend with all trading logic
- **frontend**: React frontend with Material-UI

## 🔍 Troubleshooting

### Common Issues

#### Docker Desktop Not Running
```powershell
# Start Docker Desktop manually
# Or check if it's running
docker version
```

#### Port Conflicts
If ports 80 or 8000 are in use:
```powershell
# Check what's using the ports
netstat -ano | findstr :80
netstat -ano | findstr :8000

# Stop conflicting services or change ports in docker-compose.yml
```

#### Container Won't Start
```powershell
# Check container logs
docker-compose logs backend

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

#### API Keys Not Working
1. Verify API keys in `backend/.env`
2. Check exchange API permissions
3. Ensure testnet/sandbox settings match your keys

### Performance Optimization

#### For Better Performance
```powershell
# Increase Docker resources in Docker Desktop settings
# Recommended: 4+ CPU cores, 8+ GB RAM

# Use production mode
ENVIRONMENT=production
API_WORKERS=4
```

#### For Development
```powershell
# Use development mode
ENVIRONMENT=development
API_DEBUG=true
API_RELOAD=true
```

## 🔒 Security Considerations

### Production Security
1. **Change default secrets** in `backend/.env`:
   ```env
   SECRET_KEY=your-super-secure-secret-key
   JWT_SECRET=your-super-secure-jwt-secret
   ```

2. **Use HTTPS** in production (configure reverse proxy)

3. **Restrict API access** with proper authentication

4. **Monitor logs** for suspicious activity

### API Key Security
- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly rotate API keys
- Use API keys with minimal required permissions

## 📊 Monitoring

### Health Checks
- Backend: http://localhost:8000/health
- Frontend: http://localhost:80
- Redis: Built into Docker health checks

### Logs
- Application logs: `backend/logs/`
- Docker logs: `docker-compose logs -f`
- Error logs: `backend/logs/errors_mystic_trading.log`

### Metrics
- Prometheus metrics: http://localhost:8000/metrics (if enabled)
- Performance monitoring built into the platform

## 🚀 Next Steps

After successful deployment:

1. **Test the platform**: Visit http://localhost
2. **Configure trading**: Set up your trading strategies
3. **Add API keys**: For live trading capabilities
4. **Monitor performance**: Use the built-in analytics
5. **Set up notifications**: Configure Discord/Telegram alerts

## 🆘 Support

If you encounter issues:

1. Check the logs: `docker-compose logs -f`
2. Verify environment configuration
3. Ensure Docker Desktop is running
4. Check system resources
5. Review this documentation

## ✨ Features Available

- **Real-time market data** from multiple exchanges
- **AI-powered trading signals** and decisions
- **Advanced analytics** and performance metrics
- **Social trading** features
- **Automated trading bots** with risk management
- **Portfolio management** and rebalancing
- **WebSocket real-time updates**
- **Comprehensive API** for integrations
- **Mobile-responsive dashboard**

---

**🎉 Your Mystic Trading Platform is now live and ready for cosmic analysis! 🌟**
