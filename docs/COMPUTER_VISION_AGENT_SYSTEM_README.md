# Computer Vision Agent System

## Overview

The Computer Vision Agent System provides advanced visual analysis capabilities for trading, including chart pattern recognition, technical indicator analysis, and market visualization. This system uses computer vision techniques to analyze price charts and generate trading insights.

## Architecture

### Core Components

1. **Chart Pattern Agent** - Detects technical patterns in price charts
2. **Technical Indicator Agent** - Analyzes technical indicators and generates signals
3. **Market Visualization Agent** - Generates charts and visual analysis
4. **Computer Vision Orchestrator** - Coordinates all CV agents and provides unified API

### System Flow

```
Market Data → Chart Pattern Agent → Pattern Detection
     ↓
Market Data → Technical Indicator Agent → Signal Generation
     ↓
Market Data → Market Visualization Agent → Chart Generation
     ↓
Computer Vision Orchestrator → Unified Analysis & Coordination
```

## Features

### Chart Pattern Recognition
- **Head & Shoulders** - Reversal pattern detection
- **Inverse Head & Shoulders** - Bullish reversal detection
- **Double Top/Bottom** - Reversal pattern analysis
- **Triangle Patterns** - Ascending, descending, symmetrical
- **Flag Patterns** - Bullish and bearish continuation
- **Wedge Patterns** - Rising and falling wedges
- **Channel Patterns** - Horizontal, ascending, descending

### Technical Indicator Analysis
- **Moving Averages** - SMA, EMA crossovers
- **Momentum Indicators** - RSI, Stochastic, Williams %R
- **Trend Indicators** - MACD, ADX, CCI
- **Volatility Indicators** - Bollinger Bands, ATR
- **Volume Indicators** - OBV, Volume SMA
- **Oscillators** - Parabolic SAR

### Market Visualization
- **Candlestick Charts** - Price and volume analysis
- **Line Charts** - Trend visualization
- **Volume Charts** - Volume analysis
- **Heatmaps** - Price change visualization
- **Correlation Matrices** - Asset correlation analysis

## Installation

### Prerequisites
- Docker and Docker Compose
- Redis server
- Python 3.10+

### Dependencies
```bash
# Computer Vision libraries
opencv-python==4.8.1.78
matplotlib==3.7.2
seaborn==0.12.2
Pillow==10.0.1
talib-binary==0.4.24
scipy==1.11.1
```

### Quick Start
```powershell
# Launch Computer Vision System
.\scripts\launch_computer_vision_system.ps1

# Or using Docker Compose
docker-compose up -d chart-pattern-agent technical-indicator-agent market-visualization-agent computer-vision-orchestrator
```

## Configuration

### Chart Pattern Configuration
```json
{
  "supported_patterns": [
    "head_and_shoulders",
    "inverse_head_and_shoulders",
    "double_top",
    "double_bottom",
    "triangle_ascending",
    "triangle_descending",
    "triangle_symmetrical",
    "flag_bullish",
    "flag_bearish",
    "wedge_rising",
    "wedge_falling",
    "channel_horizontal",
    "channel_ascending",
    "channel_descending"
  ],
  "confidence_threshold": 0.7,
  "min_pattern_size": 10,
  "max_pattern_size": 100
}
```

### Technical Indicator Configuration
```json
{
  "supported_indicators": [
    "sma", "ema", "rsi", "macd", "bollinger_bands",
    "stochastic", "williams_r", "cci", "adx", "atr",
    "obv", "volume_sma", "price_channels", "parabolic_sar"
  ],
  "signal_thresholds": {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "stochastic_oversold": 20,
    "stochastic_overbought": 80,
    "williams_r_oversold": -80,
    "williams_r_overbought": -20,
    "cci_oversold": -100,
    "cci_overbought": 100
  }
}
```

### Chart Generation Configuration
```json
{
  "chart_types": ["candlestick", "line", "volume", "heatmap", "correlation"],
  "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
  "indicators": ["sma", "ema", "bollinger_bands", "rsi", "macd"],
  "chart_style": "dark_background"
}
```

## API Endpoints

### Chart Pattern Agent
```python
# Analyze chart for patterns
await send_message("chart_pattern_agent", {
    "type": "analyze_chart",
    "symbol": "BTC"
})

# Detect specific patterns
await send_message("chart_pattern_agent", {
    "type": "detect_patterns",
    "symbol": "BTC",
    "patterns": ["head_and_shoulders", "double_top"]
})

# Get pattern signals
await send_message("chart_pattern_agent", {
    "type": "get_pattern_signals",
    "symbol": "BTC"
})
```

### Technical Indicator Agent
```python
# Calculate indicators
await send_message("technical_indicator_agent", {
    "type": "calculate_indicators",
    "symbol": "BTC",
    "indicators": ["rsi", "macd", "bollinger_bands"]
})

# Get indicator signals
await send_message("technical_indicator_agent", {
    "type": "get_indicator_signals",
    "symbol": "BTC"
})

# Analyze crossovers
await send_message("technical_indicator_agent", {
    "type": "analyze_crossovers",
    "symbol": "BTC"
})
```

### Market Visualization Agent
```python
# Generate chart
await send_message("market_visualization_agent", {
    "type": "generate_chart",
    "symbol": "BTC",
    "chart_type": "candlestick"
})

# Get visual analysis
await send_message("market_visualization_agent", {
    "type": "get_visual_analysis",
    "symbol": "BTC"
})
```

### Computer Vision Orchestrator
```python
# Coordinate analysis
await send_message("computer_vision_orchestrator", {
    "type": "coordinate_analysis",
    "symbol": "BTC"
})

# Get CV results
await send_message("computer_vision_orchestrator", {
    "type": "get_cv_results",
    "symbol": "BTC"
})

# Update CV configuration
await send_message("computer_vision_orchestrator", {
    "type": "update_cv_config",
    "config": {
        "agents": {
            "chart_pattern_agent": {"enabled": True},
            "technical_indicator_agent": {"enabled": True}
        }
    }
})
```

## Monitoring

### Service Status
```bash
# Check service status
docker-compose ps chart-pattern-agent technical-indicator-agent market-visualization-agent computer-vision-orchestrator

# View logs
docker-compose logs -f chart-pattern-agent
docker-compose logs -f technical-indicator-agent
docker-compose logs -f market-visualization-agent
docker-compose logs -f computer-vision-orchestrator
```

### Redis Data
```bash
# Check CV-related data
docker exec mystic-redis redis-cli keys "*cv*"
docker exec mystic-redis redis-cli keys "*pattern*"
docker exec mystic-redis redis-cli keys "*indicator*"
docker exec mystic-redis redis-cli keys "*chart*"
```

### Metrics
```bash
# View agent metrics
docker exec mystic-redis redis-cli get "agent_metrics:chart_pattern_agent_001"
docker exec mystic-redis redis-cli get "agent_metrics:technical_indicator_agent_001"
docker exec mystic-redis redis-cli get "agent_metrics:market_visualization_agent_001"
docker exec mystic-redis redis-cli get "agent_metrics:computer_vision_orchestrator_001"
```

## Integration

### With Trading System
The Computer Vision System integrates with the existing trading system:

1. **Pattern Detection** → Strategy Agent
2. **Indicator Signals** → Risk Agent
3. **Visual Analysis** → Execution Agent
4. **Unified Results** → All Agents

### Message Flow
```
Market Data → CV Agents → Pattern/Indicator/Visual Analysis
     ↓
CV Orchestrator → Aggregated Results → Trading Agents
     ↓
Trading Agents → Enhanced Trading Decisions
```

## Performance

### Optimization
- **Caching** - Chart and indicator results cached in Redis
- **Async Processing** - Non-blocking analysis operations
- **Resource Management** - Automatic cleanup of old data
- **Health Monitoring** - Continuous agent health checks

### Scalability
- **Microservice Architecture** - Independent agent scaling
- **Redis Pub/Sub** - Efficient inter-agent communication
- **Docker Containers** - Easy deployment and scaling
- **Load Balancing** - Multiple agent instances support

## Troubleshooting

### Common Issues

1. **Agent Not Starting**
   ```bash
   # Check Docker logs
   docker-compose logs chart-pattern-agent
   
   # Verify Redis connection
   docker exec mystic-redis redis-cli ping
   ```

2. **Pattern Detection Not Working**
   ```bash
   # Check market data availability
   docker exec mystic-redis redis-cli keys "*market_data*"
   
   # Verify agent configuration
   docker exec mystic-redis redis-cli get "chart_pattern_config"
   ```

3. **Indicator Calculation Errors**
   ```bash
   # Check TA-Lib installation
   docker exec mystic-technical-indicator-agent python -c "import talib; print('TA-Lib OK')"
   
   # Verify data format
   docker exec mystic-redis redis-cli get "indicator_data:BTC:*"
   ```

4. **Chart Generation Issues**
   ```bash
   # Check matplotlib backend
   docker exec mystic-market-visualization-agent python -c "import matplotlib; print(matplotlib.get_backend())"
   
   # Verify image processing libraries
   docker exec mystic-market-visualization-agent python -c "import cv2, PIL; print('OpenCV and PIL OK')"
   ```

### Debug Commands
```bash
# Enter agent containers
docker exec -it mystic-chart-pattern-agent bash
docker exec -it mystic-technical-indicator-agent bash
docker exec -it mystic-market-visualization-agent bash
docker exec -it mystic-computer-vision-orchestrator bash

# Check agent processes
docker exec mystic-chart-pattern-agent ps aux
docker exec mystic-technical-indicator-agent ps aux

# Monitor Redis traffic
docker exec mystic-redis redis-cli monitor
```

## Development

### Adding New Patterns
1. Update `pattern_templates` in `chart_pattern_agent.py`
2. Add detection logic in `detect_specific_pattern()`
3. Update configuration in Redis
4. Test with sample data

### Adding New Indicators
1. Add indicator function to `indicator_functions` in `technical_indicator_agent.py`
2. Implement calculation logic in `calculate_indicator()`
3. Add signal analysis in `analyze_*_signal()` methods
4. Update configuration

### Adding New Chart Types
1. Add chart type to `chart_config` in `market_visualization_agent.py`
2. Implement generation logic in `generate_chart_type()`
3. Add visualization settings
4. Test chart generation

## Security

### Best Practices
- **Input Validation** - Validate all market data inputs
- **Error Handling** - Comprehensive error handling and logging
- **Resource Limits** - Memory and CPU usage monitoring
- **Access Control** - Redis access control and authentication

### Monitoring
- **Health Checks** - Regular agent health monitoring
- **Performance Metrics** - CPU, memory, and response time tracking
- **Error Tracking** - Comprehensive error logging and alerting
- **Security Audits** - Regular security assessments

## Future Enhancements

### Planned Features
1. **Deep Learning Pattern Recognition** - CNN-based pattern detection
2. **Real-time Video Analysis** - Live chart video processing
3. **3D Visualization** - Multi-dimensional market analysis
4. **Advanced Pattern Recognition** - Machine learning pattern classification
5. **Custom Indicator Builder** - User-defined indicator creation
6. **Interactive Charts** - Web-based interactive visualizations

### Performance Improvements
1. **GPU Acceleration** - CUDA-based image processing
2. **Distributed Processing** - Multi-node analysis
3. **Caching Optimization** - Advanced caching strategies
4. **Memory Optimization** - Efficient memory usage
5. **Parallel Processing** - Concurrent analysis operations

## Support

### Documentation
- [API Documentation](api_documentation.md)
- [Architecture Guide](architecture.md)
- [Deployment Guide](deployment.md)
- [Troubleshooting Guide](troubleshooting.md)

### Contact
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: Project Wiki

---

**Computer Vision Agent System** - Advanced visual analysis for intelligent trading decisions. 