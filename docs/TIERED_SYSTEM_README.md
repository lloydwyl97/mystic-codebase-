# 🔧 Tiered Signal System - Mystic Trading Platform

## Overview

The Tiered Signal System implements a sophisticated three-tier architecture for cryptocurrency trading signals, designed to maximize performance and predictive power while minimizing CPU usage and API rate limits.

## 🏗️ System Architecture

### Tier 1: Real-Time Signals (5-15 seconds)

**Purpose**: Directly control autobuy/autosell decisions

| Signal | Description | Frequency | Storage |
|--------|-------------|-----------|---------|
| ✅ Price | Spot price for each coin | 5-10 sec | Redis cache |
| ✅ % Change (1m) | Momentum spikes | 10-15 sec | Cached from price diff |
| ✅ Order book | Depth/volume wall detection | 15 sec | Backend only |

**Location**: `backend/price_fetcher.py`

### Tier 2: Tactical Strategy Signals (1-5 minutes)

**Purpose**: Supports trade timing, trend strength, and decision confidence

| Signal | Description | Frequency | Storage |
|--------|-------------|-----------|---------|
| ✅ 24h Volume | Liquidity detection | 2-3 min | Backend DB |
| ✅ RSI / MACD | Trend strength or reversal | 1-3 min | Backend cache |
| ✅ % Change (5m+) | Volatility/surge tracking | 1-3 min | Cached off Tier 1 |
| ✅ Volatility Index | Detect sharp ranges | 2-5 min | Derived locally |

**Location**: `backend/indicators.py`

### Tier 3: Mystic/Cosmic/Meta Signals (30 min - 1 hour)

**Purpose**: Trend confirmation, big-picture filters

| Signal | Description | Frequency | Storage |
|--------|-------------|-----------|---------|
| ☯️ Schumann Resonance | Earth frequency match | 1 hr | Fetcher + local cache |
| ☀️ Solar Flare Index | Instability prediction (risk off) | 1 hr | NOAA API |
| 🧠 Pineal alignment | External cosmic timing signal | 1 hr+ | Static or webhook |

**Location**: `backend/cosmic_fetcher.py`

## ⚙️ Throttle Logic & Strategy Design

### Local Caching / Batching Rules

- Store last fetch timestamp for every signal
- Use check queue: If data is < valid_for period, skip
- Auto-escalate priority if trade condition gets close to triggering

### Avoid Unnecessary Loops

Do NOT fetch all indicators unless:

- Price deviation > 1-2%

- Volume spike > 20% above rolling average
- Momentum flip (negative to positive)

### Sync All Data in "Unified Model"

Keep a central coin state object per coin that holds:

- `last_price`
- `last_volume`
- `rsi`
- `macd`
- `mystic_alignment_score`
- `is_active_buy_signal`

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install redis aiohttp fastapi uvicorn
```

### 2. Start Redis Server

```bash
# Windows
redis-server

# Linux/Mac
redis-server /etc/redis/redis.conf
```

### 3. Run the Tiered System

```bash
# Demo mode (5 minutes)
python backend/start_tiered_system.py demo

# Continuous mode
python backend/start_tiered_system.py continuous
```

### 4. Start the Backend API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 API Endpoints

### Tier 1 Signals

```http
GET /api/signals/tier1
GET /api/signals/tier1?symbol=BTCUSDT
```

### Tier 2 Signals

```http
GET /api/signals/tier2
GET /api/signals/tier2?symbol=BTCUSDT
```

### Tier 3 Signals

```http
GET /api/signals/tier3
```

### Unified Signals

```http
GET /api/signals/unified
GET /api/signals/unified?symbol=BTCUSDT
```

### Trade Decisions

```http
GET /api/signals/trade-decisions
```

### System Status

```http
GET /api/signals/summary
GET /api/signals/status
```

### Control Endpoints

```http
POST /api/signals/start
POST /api/signals/stop
```

## 🔧 Configuration

### Default Configuration

```python
from backend.tiered_system_config import get_config

# Get default configuration
config = get_config("default")

# Get optimized configuration for high-frequency trading
config = get_config("optimized")

# Get conservative configuration for lower resource usage
config = get_config("conservative")
```

### Custom Configuration

```python
from backend.tiered_system_config import TieredSystemConfig

config = TieredSystemConfig()

# Customize Tier 1 settings
config.tier1.price_fetch_interval = 3  # 3 seconds
config.tier1.binance_coins = ['BTCUSDT', 'ETHUSDT']

# Customize Tier 2 settings
config.tier2.rsi_period = 21
config.tier2.rsi_oversold = 25

# Customize Tier 3 settings
config.tier3.schumann_fetch_interval = 1800  # 30 minutes
```

## 📈 Performance Optimization

### High-Frequency Trading Setup

```python
# Use optimized configuration
config = get_config("optimized")

# Key optimizations:
# - Tier 1: 3-5 second intervals
# - Tier 2: 1-2 minute intervals
# - Tier 3: 30 minute intervals
# - Trade decisions: 3 second intervals
```

### Conservative Setup

```python
# Use conservative configuration
config = get_config("conservative")

# Key settings:
# - Tier 1: 10-30 second intervals
# - Tier 2: 5-10 minute intervals
# - Tier 3: 2-4 hour intervals
# - Trade decisions: 10 second intervals
```

## 🧠 Signal Processing Logic

### Signal Strength Calculation

```python
# Tier 1: Price momentum
momentum = abs(price_change_1m)
if momentum > 5: strength += 3
elif momentum > 2: strength += 2
elif momentum > 1: strength += 1

# Tier 2: Technical indicators
if rsi < 20 or rsi > 80: strength += 3
elif rsi < 30 or rsi > 70: strength += 2

if abs(macd_histogram) > 0.01: strength += 2
elif abs(macd_histogram) > 0.005: strength += 1

# Tier 3: Cosmic alignment
if cosmic_score > 80: strength += 2
elif cosmic_score > 60: strength += 1
```

### Confidence Calculation

```python
# Price stability (Tier 1)
confidence += 0.8

# Technical agreement (Tier 2)
if rsi_bullish == macd_bullish: confidence += 0.9
else: confidence += 0.6

# Cosmic alignment (Tier 3)
if cosmic_score > 70: confidence += 0.85
elif cosmic_score > 50: confidence += 0.7
else: confidence += 0.5
```

## 🔍 Monitoring & Debugging

### Health Check

```bash
curl http://localhost:8000/api/signals/summary
```

### Log Monitoring

```bash
tail -f tiered_system.log
```

### Redis Cache Inspection

```bash
redis-cli
> KEYS *tier*
> GET tier1_signals
> GET unified_signals
```

## 🚨 Error Handling

### Common Issues

1. **Redis Connection Failed**

   ```Text
   Error: Failed to connect to Redis
   Solution: Start Redis server or check connection settings
   ```

2. **API Rate Limits**

   ``Text`
   Error: 429 Too Many Requests
   Solution: Increase fetch intervals in configuration

3. **Missing Dependencies**

   ```Text
   Error: ModuleNotFoundError
   Solution: Install required packages with pip
   ```

### Recovery Procedures

1. **Automatic Restart**
   - Components automatically restart on failure
   - Maximum 3 restart attempts per component

2. **Manual Recovery**

   ```bash
   # Stop all components
   curl -X POST http://localhost:8000/api/signals/stop

   # Start fresh
   curl -X POST http://localhost:8000/api/signals/start
   ```

## 📊 Metrics & Analytics

### Performance Metrics

- Signal generation latency
- Cache hit/miss ratios
- API response times
- Trade decision accuracy

### System Health

- Component status
- Memory usage
- CPU utilization
- Network I/O

## 🔮 Future Enhancements

### Planned Features

- Machine learning signal integration
- Advanced pattern recognition
- Multi-exchange arbitrage signals
- Portfolio optimization algorithms

### Scalability Improvements

- Horizontal scaling with multiple instances
- Load balancing across signal processors
- Distributed caching with Redis Cluster
- Microservices architecture

## 📚 Additional Resources

### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Redis Documentation](https://redis.io/documentation)
- [aiohttp Documentation](https://docs.aiohttp.org/)

### Trading Resources

- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Coinbase API Documentation](https://docs.pro.coinbase.us/)

### Cosmic/Mystic Resources

- [NOAA Space Weather](https://www.swpc.noaa.gov/)
- [Schumann Resonance](https://www2.irf.se/maggraphs/schumann)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This system is designed for educational and research purposes. Always test thoroughly before using with real trading funds
