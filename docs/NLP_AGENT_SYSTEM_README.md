# 🧠 NLP Agent System

## Overview

The **NLP Agent System** is a comprehensive natural language processing framework that provides real-time sentiment analysis for cryptocurrency trading decisions. It integrates news sentiment, social media monitoring, and market sentiment aggregation to deliver unified sentiment signals to the trading system.

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    NLP Agent System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ News Sentiment  │    │ Social Media    │                │
│  │     Agent       │    │     Agent       │                │
│  │                 │    │                 │                │
│  │ • RSS Feeds     │    │ • Twitter       │                │
│  │ • News APIs     │    │ • Reddit        │                │
│  │ • Sentiment     │    │ • Telegram      │                │
│  │   Analysis      │    │ • Trending      │                │
│  └─────────────────┘    │   Topics        │                │
│                         └─────────────────┘                │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Market Sentiment│    │ NLP Orchestrator│                │
│  │     Agent       │    │                 │                │
│  │                 │    │ • Coordination  │                │
│  │ • Aggregation   │    │ • Service Mgmt  │                │
│  │ • Fear & Greed  │    │ • Unified API   │                │
│  │ • Signal Gen    │    │ • Health Check  │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Trading System                          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Strategy    │  │ Risk        │  │ Execution   │        │
│  │ Agent       │  │ Agent       │  │ Agent       │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **News Sentiment Agent** fetches and analyzes financial news
2. **Social Media Agent** monitors social platforms for sentiment
3. **Market Sentiment Agent** aggregates all sentiment sources
4. **NLP Orchestrator** coordinates and provides unified API
5. **Trading Agents** receive sentiment signals for decision making

## 🚀 Features

### News Sentiment Agent
- **Real-time News Analysis**: Monitors RSS feeds from major financial news sources
- **Sentiment Extraction**: Uses TextBlob for sentiment analysis
- **Symbol Detection**: Automatically identifies trading symbols in news articles
- **Keyword Analysis**: Extracts sentiment for specific financial keywords
- **Source Management**: Configurable news sources and keywords

### Social Media Agent
- **Multi-Platform Monitoring**: Twitter, Reddit, Telegram support
- **Trending Topics**: Identifies trending hashtags and topics
- **Influencer Tracking**: Monitors key influencer sentiment
- **Engagement Scoring**: Calculates post engagement metrics
- **Emoji Sentiment**: Analyzes emoji sentiment in social posts

### Market Sentiment Agent
- **Sentiment Aggregation**: Combines sentiment from all sources
- **Fear & Greed Index**: Calculates market sentiment index
- **Signal Generation**: Creates trading signals based on sentiment
- **Source Weighting**: Configurable weights for different sources
- **Historical Analysis**: Maintains sentiment history for trends

### NLP Orchestrator
- **Agent Coordination**: Manages all NLP agent communication
- **Service Management**: Provides unified NLP services
- **Health Monitoring**: Monitors agent health and status
- **Configuration Management**: Centralized configuration control
- **API Gateway**: Single entry point for NLP services

## 📊 Key Metrics

### Sentiment Metrics
- **Polarity**: Sentiment score (-1 to +1)
- **Subjectivity**: How subjective vs objective the content is
- **Confidence**: Confidence level in sentiment analysis
- **Engagement**: Social media engagement metrics
- **Fear & Greed**: Market sentiment index (0-100)

### Performance Metrics
- **Analysis Count**: Number of articles/posts analyzed
- **Response Time**: Time to process sentiment requests
- **Accuracy**: Sentiment analysis accuracy
- **Coverage**: Number of sources and symbols monitored
- **Uptime**: System availability and reliability

## 🛠️ Installation & Deployment

### Prerequisites
- Docker and Docker Compose
- Redis server
- Python 3.10+
- Required Python packages (see requirements.txt)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Mystic-Codebase
   ```

2. **Launch the NLP system**:
   ```powershell
   .\scripts\launch_nlp_system.ps1
   ```

3. **Verify deployment**:
   ```bash
   docker ps --filter name=mystic-
   ```

### Manual Deployment

1. **Start Redis**:
   ```bash
   docker-compose up -d redis
   ```

2. **Start NLP services**:
   ```bash
   docker-compose up -d news-sentiment-agent
   docker-compose up -d social-media-agent
   docker-compose up -d market-sentiment-agent
   docker-compose up -d nlp-orchestrator
   ```

3. **Check service health**:
   ```bash
   docker logs mystic-nlp-orchestrator
   ```

## 🔧 Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://redis:6379

# Service Configuration
SERVICE_TYPE=nlp_orchestrator
SERVICE_PORT=8000

# NLP Configuration
NEWS_SOURCES_CONFIG=news_sources.json
SOCIAL_PLATFORMS_CONFIG=social_platforms.json
SENTIMENT_WEIGHTS_CONFIG=sentiment_weights.json
```

### Configuration Files

#### News Sources Configuration
```json
{
  "sources": [
    {
      "name": "Reuters Business",
      "url": "http://feeds.reuters.com/reuters/businessNews",
      "type": "rss",
      "keywords": ["crypto", "bitcoin", "ethereum", "blockchain"]
    }
  ]
}
```

#### Social Platforms Configuration
```json
{
  "platforms": [
    {
      "name": "Twitter",
      "type": "twitter",
      "keywords": ["#bitcoin", "#crypto", "#btc"],
      "enabled": true
    }
  ]
}
```

#### Sentiment Weights Configuration
```json
{
  "weights": {
    "news_sentiment": 0.3,
    "social_media": 0.25,
    "market_data": 0.25,
    "technical_indicators": 0.2
  }
}
```

## 📡 API Reference

### NLP Orchestrator API

#### Get Unified Sentiment
```http
GET /api/nlp/sentiment/{symbol}
```

Response:
```json
{
  "symbol": "BTC",
  "unified_sentiment": {
    "polarity": 0.45,
    "category": "positive",
    "confidence": 0.78,
    "source_contributions": {
      "news": {"avg_polarity": 0.3, "count": 15},
      "social": {"avg_polarity": 0.6, "count": 25},
      "market": {"avg_polarity": 0.4, "count": 10}
    }
  }
}
```

#### Get Fear & Greed Index
```http
GET /api/nlp/fear-greed
```

Response:
```json
{
  "score": 65,
  "category": "Greed",
  "avg_polarity": 0.25,
  "symbols_count": 7,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Get Trending Topics
```http
GET /api/nlp/trending
```

Response:
```json
{
  "hashtags": {
    "#bitcoin": 150,
    "#crypto": 120,
    "#ethereum": 85
  },
  "keywords": {
    "bullish": 45,
    "rally": 32
  }
}
```

### Agent Communication

#### Send Message to Agent
```python
await orchestrator.send_message("news_sentiment_agent_001", {
    "type": "analyze_news",
    "source_url": "https://example.com/news",
    "keywords": ["bitcoin", "crypto"]
})
```

#### Broadcast Message
```python
await orchestrator.broadcast_message({
    "type": "sentiment_update",
    "symbol": "BTC",
    "sentiment": {"polarity": 0.5, "category": "positive"}
})
```

## 🔍 Monitoring & Debugging

### Health Checks

```bash
# Check all NLP services
docker ps --filter name=mystic-

# Check specific service logs
docker logs mystic-news-sentiment-agent
docker logs mystic-social-media-agent
docker logs mystic-market-sentiment-agent
docker logs mystic-nlp-orchestrator

# Check Redis status
docker exec mystic-redis redis-cli ping
```

### Metrics Dashboard

Access metrics via Redis:
```bash
# Get agent metrics
docker exec mystic-redis redis-cli get "agent_metrics:nlp_orchestrator_001"

# Get sentiment data
docker exec mystic-redis redis-cli get "unified_sentiment:BTC"

# Get fear & greed index
docker exec mystic-redis redis-cli get "market_fear_greed_index"
```

### Troubleshooting

#### Common Issues

1. **Agent not starting**:
   - Check Docker logs: `docker logs <container-name>`
   - Verify Redis connection
   - Check environment variables

2. **No sentiment data**:
   - Verify news sources are accessible
   - Check social media API keys
   - Monitor agent communication

3. **High latency**:
   - Check Redis performance
   - Monitor system resources
   - Review agent processing loops

## 🔄 Integration

### Trading System Integration

The NLP system integrates seamlessly with the existing trading system:

1. **Strategy Agent**: Receives sentiment signals for strategy decisions
2. **Risk Agent**: Uses sentiment for risk assessment
3. **Execution Agent**: Considers sentiment for trade timing
4. **Compliance Agent**: Monitors sentiment for compliance alerts

### Data Flow Integration

```
News/Social Data → NLP Agents → Sentiment Analysis → 
Aggregation → Trading Signals → Strategy/Risk Agents → 
Trading Decisions
```

## 🚀 Advanced Features

### Custom Sentiment Models
- Replace TextBlob with custom models (FinBERT, VADER)
- Fine-tune models on crypto-specific data
- Implement ensemble methods for better accuracy

### Real-time Streaming
- WebSocket connections for real-time updates
- Kafka integration for high-throughput data
- Event-driven architecture for scalability

### Machine Learning Integration
- Sentiment forecasting models
- Trend prediction algorithms
- Anomaly detection in sentiment patterns

## 📈 Performance Optimization

### Scaling Strategies
- Horizontal scaling of NLP agents
- Load balancing across multiple instances
- Caching strategies for frequently accessed data

### Resource Management
- Memory optimization for large datasets
- CPU utilization monitoring
- Network bandwidth optimization

## 🔒 Security & Compliance

### Data Privacy
- Anonymization of social media data
- Secure storage of sentiment analysis results
- Compliance with data protection regulations

### Access Control
- API authentication and authorization
- Rate limiting for external APIs
- Audit logging for all operations

## 📚 Development

### Adding New Sources

1. **Create source configuration**:
   ```python
   new_source = {
       "name": "New Source",
       "url": "https://source.com/feed",
       "type": "rss",
       "keywords": ["crypto", "trading"]
   }
   ```

2. **Update agent configuration**:
   ```python
   await agent.update_sources([new_source])
   ```

### Custom Sentiment Analysis

1. **Implement custom analyzer**:
   ```python
   class CustomSentimentAnalyzer:
       def analyze(self, text):
           # Custom sentiment logic
           return {"polarity": 0.5, "confidence": 0.8}
   ```

2. **Register with agent**:
   ```python
   agent.set_sentiment_analyzer(CustomSentimentAnalyzer())
   ```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Testing
```bash
# Run unit tests
python -m pytest tests/test_nlp_agents.py

# Run integration tests
python -m pytest tests/test_nlp_integration.py

# Run performance tests
python -m pytest tests/test_nlp_performance.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review troubleshooting guide
- Contact the development team

---

**NLP Agent System** - Powering intelligent trading decisions with real-time sentiment analysis 🧠📊 