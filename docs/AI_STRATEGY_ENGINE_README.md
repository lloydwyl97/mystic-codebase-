# 🧠 Enhanced AI Strategy Engine

## Overview

The Enhanced AI Strategy Engine is a comprehensive artificial intelligence system for automated trading strategy generation, evolution, and optimization. Built with advanced neural networks, genetic algorithms, and real-time monitoring capabilities.

## 🏗️ Architecture

### Core Components

1. **AI Strategy Generator** (`ai_strategy_generator.py`)
   - Neural network-based strategy generation
   - LSTM and Transformer models
   - Real-time market data processing
   - Automated strategy creation

2. **Genetic Algorithm Engine** (`ai_genetic_algorithm.py`)
   - Population-based strategy evolution
   - Fitness-based selection and crossover
   - Parameter optimization
   - Multi-generational improvement

3. **Model Versioning Service** (`ai_model_versioning.py`)
   - Version control for AI models
   - Performance tracking
   - Model lineage management
   - Automated deployment

4. **Auto-Retrain Service** (`ai_auto_retrain.py`)
   - Performance monitoring
   - Automatic retraining triggers
   - Model improvement detection
   - Continuous optimization

5. **Mutation Dashboard** (`ai_mutation_dashboard.py`)
   - Real-time monitoring interface
   - Strategy performance visualization
   - Manual control and intervention
   - System health monitoring

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Windows 11 with PowerShell
- At least 8GB RAM
- 20GB free disk space

### Installation

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd Mystic-Codebase
   ```

2. **Launch the enhanced system**
   ```powershell
   .\scripts\launch-ai-enhanced-system.ps1
   ```

3. **Access the dashboards**
   - Main Dashboard: http://localhost:8501
   - AI Mutation Dashboard: http://localhost:8080
   - Backend API: http://localhost:8000

## 📊 System Services

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend Dashboard | 8501 | Main trading interface |
| Backend API | 8000 | Core API endpoints |
| AI Strategy Generator | 8002 | Strategy generation service |
| AI Genetic Algorithm | 8003 | Evolution engine |
| AI Model Versioning | 8004 | Model management |
| AI Auto-Retrain | 8005 | Retraining service |
| AI Mutation Dashboard | 8080 | AI monitoring interface |
| Redis | 6379 | Cache and messaging |

### Service Dependencies

```
Redis (Core)
├── Backend API
├── Frontend Dashboard
├── AI Strategy Generator
├── AI Genetic Algorithm
├── AI Model Versioning
├── AI Auto-Retrain
└── AI Mutation Dashboard
```

## 🧠 AI Components Deep Dive

### 1. AI Strategy Generator

**Purpose**: Generate new trading strategies using neural networks

**Features**:
- LSTM networks for time series prediction
- Transformer models for market analysis
- Real-time data processing
- Multi-symbol support
- Automated parameter tuning

**Configuration**:
```python
model_configs = {
    'lstm': {
        'input_size': 10,
        'hidden_size': 128,
        'num_layers': 3,
        'output_size': 3,  # Buy, Hold, Sell
        'sequence_length': 60
    },
    'transformer': {
        'input_size': 10,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'output_size': 3,
        'sequence_length': 60
    }
}
```

### 2. Genetic Algorithm Engine

**Purpose**: Evolve and optimize trading strategies

**Features**:
- Population-based evolution
- Tournament selection
- Crossover and mutation operators
- Fitness-based optimization
- Multi-objective optimization

**Parameters**:
```python
parameter_ranges = {
    'rsi_period': (10, 30),
    'rsi_oversold': (20, 40),
    'rsi_overbought': (60, 80),
    'sma_short': (5, 25),
    'sma_long': (20, 100),
    'macd_fast': (8, 16),
    'macd_slow': (20, 32),
    'macd_signal': (5, 15),
    'bb_period': (10, 30),
    'bb_std': (1.5, 3.0),
    'volume_sma': (10, 30),
    'stop_loss': (0.02, 0.10),
    'take_profit': (0.05, 0.20),
    'position_size': (0.05, 0.25),
    'max_positions': (1, 5)
}
```

### 3. Model Versioning Service

**Purpose**: Track and manage AI model versions

**Features**:
- Version control for models
- Performance tracking
- Model lineage
- Automated deployment
- Rollback capabilities

**Database Schema**:
```sql
-- Model versions table
CREATE TABLE model_versions (
    version_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    metadata TEXT,
    performance_metrics TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    status TEXT DEFAULT 'ACTIVE',
    parent_version TEXT,
    tags TEXT,
    description TEXT,
    file_hash TEXT,
    file_size INTEGER
);

-- Model lineage table
CREATE TABLE model_lineage (
    version_id TEXT,
    parent_version TEXT,
    relationship_type TEXT,
    created_at TEXT,
    PRIMARY KEY (version_id, parent_version)
);

-- Model deployments table
CREATE TABLE model_deployments (
    deployment_id TEXT PRIMARY KEY,
    version_id TEXT,
    environment TEXT,
    deployed_at TEXT,
    status TEXT,
    performance_snapshot TEXT,
    FOREIGN KEY (version_id) REFERENCES model_versions (version_id)
);
```

### 4. Auto-Retrain Service

**Purpose**: Automatically retrain models when performance degrades

**Features**:
- Performance monitoring
- Degradation detection
- Automatic retraining
- Model comparison
- Quality assurance

**Retraining Triggers**:
- 5% accuracy degradation
- 2% return degradation
- 24-hour time interval
- Manual trigger

### 5. Mutation Dashboard

**Purpose**: Monitor and control AI strategy evolution

**Features**:
- Real-time system monitoring
- Strategy performance visualization
- Manual intervention controls
- Evolution history tracking
- Service health monitoring

## 🔧 Configuration

### Environment Variables

Create `.env` files in each service directory:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# AI Service Configuration
SERVICE_PORT=8002
MODEL_STORAGE_PATH=./models
SCALER_STORAGE_PATH=./scalers
VERSION_STORAGE_PATH=./model_versions

# Training Configuration
BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=50
SEQUENCE_LENGTH=60

# Genetic Algorithm Configuration
POPULATION_SIZE=50
MUTATION_RATE=0.1
CROSSOVER_RATE=0.8
ELITE_SIZE=5
MAX_GENERATIONS=100
FITNESS_THRESHOLD=0.8

# Retraining Configuration
RETRAIN_THRESHOLD=0.05
RETRAIN_INTERVAL_HOURS=24
PERFORMANCE_WINDOW_DAYS=7
MIN_DATA_POINTS=1000
```

### Docker Configuration

The system uses Docker Compose for orchestration:

```yaml
services:
  ai-strategy-generator:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: mystic-ai-strategy-generator
    ports:
      - "8002:8002"
    env_file:
      - backend/.env
    volumes:
      - ./backend:/app
      - ./backend/models:/app/models
      - ./backend/scalers:/app/scalers
    command: python ai_strategy_generator.py
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
```

## 📈 Performance Metrics

### Strategy Performance

- **Accuracy**: Prediction accuracy percentage
- **Total Return**: Cumulative return over time
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### System Performance

- **Model Generation Rate**: Strategies per hour
- **Evolution Speed**: Generations per day
- **Retraining Frequency**: Models retrained per day
- **System Uptime**: Service availability percentage
- **Response Time**: API response latency

## 🔍 Monitoring and Debugging

### Health Checks

Each service includes health check endpoints:

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8005/health
curl http://localhost:8080/api/status
```

### Logs

View service logs:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ai-strategy-generator
docker-compose logs -f ai-genetic-algorithm
docker-compose logs -f ai-mutation-dashboard
```

### Metrics

Monitor system metrics:

```bash
# Redis metrics
docker exec mystic-redis redis-cli info

# Service status
docker-compose ps

# Resource usage
docker stats
```

## 🛠️ Development

### Adding New AI Models

1. **Create model class**:
```python
class NewAIModel(nn.Module):
    def __init__(self, config):
        super(NewAIModel, self).__init__()
        # Model architecture
        
    def forward(self, x):
        # Forward pass
        return output
```

2. **Add to configuration**:
```python
model_configs['new_model'] = {
    'input_size': 10,
    'hidden_size': 128,
    'output_size': 3,
    'sequence_length': 60
}
```

3. **Update training logic**:
```python
if strategy_type == 'new_model':
    model = NewAIModel(config)
```

### Customizing Genetic Algorithm

1. **Add new parameters**:
```python
parameter_ranges['new_param'] = (min_val, max_val)
```

2. **Modify fitness function**:
```python
def calculate_fitness(self, performance):
    # Custom fitness calculation
    return fitness_score
```

3. **Add new genetic operators**:
```python
def custom_crossover(self, parent1, parent2):
    # Custom crossover logic
    return child
```

## 🚨 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check Docker status
   docker info
   
   # Check service logs
   docker-compose logs [service-name]
   
   # Restart services
   docker-compose restart
   ```

2. **Redis connection issues**
   ```bash
   # Test Redis connectivity
   docker exec mystic-redis redis-cli ping
   
   # Check Redis logs
   docker-compose logs redis
   ```

3. **Model training failures**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Check memory usage
   docker stats
   
   # Verify data paths
   ls -la backend/models/
   ```

4. **Performance degradation**
   ```bash
   # Check system resources
   docker stats
   
   # Monitor Redis memory
   docker exec mystic-redis redis-cli info memory
   
   # Review recent logs
   docker-compose logs --tail=100
   ```

### Performance Optimization

1. **GPU Acceleration**
   ```yaml
   # Add GPU support to Docker Compose
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

2. **Memory Optimization**
   ```python
   # Batch processing
   batch_size = 64
   
   # Gradient accumulation
   accumulation_steps = 4
   
   # Mixed precision training
   scaler = GradScaler()
   ```

3. **Redis Optimization**
   ```bash
   # Increase memory limit
   redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
   
   # Enable persistence
   redis-server --save 900 1 --save 300 10 --save 60 10000
   ```

## 📚 API Documentation

### AI Strategy Generator API

```python
# Generate new strategy
POST /api/strategies/generate
{
    "type": "lstm",
    "symbol": "BTC/USDT",
    "parameters": {
        "learning_rate": 0.001,
        "epochs": 50
    }
}

# Get strategy details
GET /api/strategies/{strategy_id}

# List all strategies
GET /api/strategies
```

### Genetic Algorithm API

```python
# Get population
GET /api/genetic-population

# Trigger evolution
POST /api/genetic/evolve

# Reset population
POST /api/genetic/reset

# Get evolution history
GET /api/evolution-history
```

### Model Versioning API

```python
# Get model versions
GET /api/model-versions

# Deploy model
POST /api/model-versions/{version_id}/deploy

# Compare versions
GET /api/model-versions/compare/{version1}/{version2}

# Get model lineage
GET /api/model-versions/{version_id}/lineage
```

### Auto-Retrain API

```python
# Get retrain queue
GET /api/retrain-queue

# Trigger retrain
POST /api/strategies/{strategy_id}/retrain

# Get retrain history
GET /api/retrain-history
```

## 🔮 Future Enhancements

### Planned Features

1. **Advanced Neural Networks**
   - Attention mechanisms
   - Graph neural networks
   - Reinforcement learning integration

2. **Enhanced Genetic Algorithm**
   - Multi-objective optimization
   - Adaptive mutation rates
   - Island model evolution

3. **Real-time Data Integration**
   - Live market data feeds
   - News sentiment analysis
   - Social media sentiment

4. **Advanced Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert systems

5. **Model Interpretability**
   - SHAP explanations
   - Feature importance analysis
   - Decision tree visualization

### Scalability Improvements

1. **Distributed Training**
   - Multi-GPU training
   - Distributed data parallel
   - Model parallelism

2. **Microservices Architecture**
   - Service mesh
   - Load balancing
   - Auto-scaling

3. **Cloud Integration**
   - Kubernetes deployment
   - Cloud storage
   - Managed services

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
- Contact the development team

---

**Note**: This system is designed for educational and research purposes. Always test thoroughly before using in production trading environments. 