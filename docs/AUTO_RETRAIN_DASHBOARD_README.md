# 🔄 Auto-Retrain Dashboard System

## Overview

The Auto-Retrain Dashboard provides real-time monitoring and control of AI model retraining processes. It offers comprehensive visibility into model performance, retrain queues, version management, and system health.

## 🎯 Features

### 📊 Live Model Performance Monitoring
- Real-time accuracy, return, and Sharpe ratio tracking
- Performance trend analysis (improving/declining/stable)
- Model status indicators (active/inactive)
- Last retrain timestamps

### 🔄 Retrain Controls
- Manual retrain triggers with priority levels
- Retrain queue management
- Real-time retrain progress tracking
- Estimated completion times

### 📈 Retrain History & Analytics
- Comprehensive retrain history
- Performance improvement tracking
- Success rate analytics
- Duration and efficiency metrics
- Interactive charts and visualizations

### 📦 Model Version Management
- Model version timeline
- Version comparison tools
- Performance tracking across versions
- Deployment status monitoring

### 🏥 System Health Monitoring
- Service uptime tracking
- Alert management
- Performance metrics
- System status indicators

## 🏗️ Architecture

### Frontend Components
```
frontend/pages/auto_retrain_dashboard.py
├── Live Model Performance
├── Retrain Controls
├── Retrain History & Analytics
├── Model Version Management
└── System Health Monitoring
```

### Backend Services
```
backend/
├── ai_auto_retrain.py          # Auto-retrain service
├── ai_model_versioning.py      # Model versioning service
├── ai_strategy_generator.py    # Strategy generation
└── genetic_algorithm_engine.py # Genetic algorithm engine
```

### Redis Channels
- `model_metrics` - Live model performance data
- `retrain_status` - Retrain queue and progress
- `model_versions` - Model version updates
- `ai_strategies` - Active AI strategies

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- PowerShell (Windows)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Mystic-Codebase
   ```

2. **Launch the system**
   ```powershell
   .\scripts\launch_auto_retrain_dashboard.ps1
   ```

3. **Access the dashboard**
   - Open browser: http://localhost:8502
   - Backend API: http://localhost:8000

## 📱 Dashboard Interface

### Main Dashboard Sections

#### 1. Live Model Performance
- **Model Cards**: Individual model performance metrics
- **Performance Indicators**: Color-coded accuracy, returns, Sharpe ratios
- **Status Tracking**: Active/inactive status with trend indicators
- **Last Retrain**: Timestamp of most recent retrain

#### 2. Retrain Controls
- **Manual Trigger**: Select model, reason, and priority
- **Queue Management**: View and manage retrain queue
- **Progress Tracking**: Real-time retrain progress
- **Status Updates**: Current retrain status and ETA

#### 3. Retrain History & Analytics
- **Overview Tab**: Key metrics and recent performance
- **Performance Trends**: Time-series analysis
- **Detailed History**: Complete retrain log with filters

#### 4. Model Version Management
- **Version Timeline**: Visual version history
- **Performance Comparison**: Version-to-version analysis
- **Deployment Status**: Active/archived version tracking

#### 5. System Health
- **Service Status**: Uptime and health indicators
- **Alert Management**: System alerts and notifications
- **Performance Metrics**: Overall system performance

## 🔧 Configuration

### Environment Variables
```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Backend Configuration
BACKEND_URL=http://localhost:8000

# Dashboard Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Docker Services
```yaml
# Core Services
redis:                    # Redis database
backend:                  # Backend API server
enhanced-frontend:        # Auto-retrain dashboard

# AI Services
ai-auto-retrain:          # Auto-retrain service
ai-model-versioning:      # Model versioning service
ai-strategy-generator:    # Strategy generation
genetic-algorithm-engine: # Genetic algorithm engine
```

## 📊 Data Flow

### Real-time Updates
1. **Model Performance**: AI services update Redis with metrics
2. **Retrain Status**: Auto-retrain service broadcasts status changes
3. **Version Updates**: Model versioning service publishes version changes
4. **Dashboard**: Streamlit dashboard polls Redis for updates

### WebSocket Broadcasting
```python
# Model metrics broadcast
redis_client.publish('model_metrics', json.dumps(metrics_payload))

# Retrain status broadcast
redis_client.publish('retrain_status', json.dumps(status_payload))

# Model versions broadcast
redis_client.publish('model_versions', json.dumps(versions_payload))
```

## 🎛️ Usage Guide

### Monitoring Model Performance

1. **View Live Metrics**
   - Open dashboard at http://localhost:8502
   - Check "Live Model Performance" section
   - Monitor accuracy, returns, and Sharpe ratios

2. **Identify Performance Issues**
   - Look for declining performance trends
   - Check accuracy drops below thresholds
   - Monitor win rate changes

### Triggering Manual Retrains

1. **Select Model**
   - Choose from dropdown in "Retrain Controls"
   - Verify current performance metrics

2. **Set Retrain Parameters**
   - Select retrain reason (performance_degradation, manual, etc.)
   - Set priority level (low, medium, high, urgent)

3. **Monitor Progress**
   - Watch retrain queue for your model
   - Track progress in real-time
   - Check estimated completion time

### Analyzing Retrain History

1. **View Overview**
   - Check total retrains and success rates
   - Monitor average duration and improvements

2. **Analyze Trends**
   - Use "Performance Trends" tab
   - Identify patterns in retrain success
   - Track performance improvements over time

3. **Detailed Analysis**
   - Use "Detailed History" tab
   - Filter by model, date, or reason
   - Export data for external analysis

### Managing Model Versions

1. **Version Timeline**
   - View all model versions chronologically
   - Compare performance across versions
   - Track deployment status

2. **Version Comparison**
   - Compare two versions side-by-side
   - Analyze performance differences
   - Review metadata and parameters

## 🔍 Troubleshooting

### Common Issues

#### Dashboard Not Loading
```bash
# Check if services are running
docker-compose ps

# Check service logs
docker-compose logs enhanced-frontend
docker-compose logs backend
docker-compose logs redis
```

#### No Real-time Updates
```bash
# Check Redis connection
docker-compose exec redis redis-cli ping

# Check Redis channels
docker-compose exec redis redis-cli pubsub channels
```

#### Retrain Not Triggering
```bash
# Check auto-retrain service
docker-compose logs ai-auto-retrain

# Check retrain queue
docker-compose exec redis redis-cli lrange retrain_queue 0 -1
```

### Performance Optimization

#### Redis Optimization
```bash
# Monitor Redis memory usage
docker-compose exec redis redis-cli info memory

# Check Redis performance
docker-compose exec redis redis-cli info stats
```

#### Dashboard Performance
- Use caching for expensive operations
- Implement pagination for large datasets
- Optimize database queries

## 📈 Metrics & KPIs

### Key Performance Indicators

#### Model Performance
- **Accuracy**: Target > 75%
- **Total Return**: Target > 15%
- **Sharpe Ratio**: Target > 1.2
- **Win Rate**: Target > 60%

#### Retrain Efficiency
- **Success Rate**: Target > 90%
- **Average Duration**: Target < 60 minutes
- **Improvement Rate**: Target > 5% per retrain

#### System Health
- **Uptime**: Target > 99.5%
- **Response Time**: Target < 2 seconds
- **Error Rate**: Target < 1%

## 🔮 Future Enhancements

### Planned Features
- **A/B Testing**: Compare model versions in production
- **Automated Alerts**: Email/SMS notifications for issues
- **Advanced Analytics**: Machine learning insights
- **API Integration**: External model management
- **Multi-Environment**: Staging/production environments

### Roadmap
- **Phase 2B**: Enhanced UI and real-time features
- **Phase 3**: Multi-agent AI system
- **Phase 4**: Enterprise features and API marketplace

## 📚 API Reference

### Backend Endpoints

#### Model Management
```http
GET /api/models                    # Get all models
GET /api/models/{model_id}         # Get specific model
POST /api/models/{model_id}/retrain # Trigger retrain
```

#### Version Management
```http
GET /api/versions                  # Get all versions
GET /api/versions/{version_id}     # Get specific version
POST /api/versions                 # Create new version
```

#### Analytics
```http
GET /api/analytics/performance     # Performance metrics
GET /api/analytics/retrain-history # Retrain history
GET /api/analytics/system-health   # System health
```

## 🤝 Contributing

### Development Setup
1. Clone repository
2. Install dependencies
3. Set up environment variables
4. Run development server

### Code Standards
- Follow PEP 8 for Python
- Use type hints
- Add comprehensive docstrings
- Write unit tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub
- Contact the development team

---

**🎯 Auto-Retrain Dashboard System** - Monitor, control, and optimize your AI trading models in real-time. 