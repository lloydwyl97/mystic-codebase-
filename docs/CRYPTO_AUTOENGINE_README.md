# CRYPTO AUTOENGINE

A complete cryptocurrency trading automation system with 40 coins (20 Coinbase + 20 Binance), 65+ P-wise strategies, and cosmic/mystic data integration.

## 🎯 System Overview

The CRYPTO AUTOENGINE is a comprehensive trading automation platform that:

- **Monitors 40 coins** (20 Coinbase + 20 Binance US)
- **Runs 65+ strategies** per coin using cached data
- **Executes automated trades** based on strategy signals
- **Integrates cosmic/mystic data** for enhanced decision making
- **Minimizes API calls** through intelligent caching and throttling
- **Provides real-time frontend updates** every 5-10 seconds

## 🏗️ Architecture

### Core Components

1. **Configuration System** (`crypto_autoengine_config.py`)
   - Central configuration for all 40 coins
   - Fetcher intervals and throttling rules
   - Strategy parameters and API settings

2. **Shared Cache System** (`shared_cache.py`)
   - Per-coin cache with price history
   - Redis integration for persistence
   - Smart update logic with change thresholds

3. **Data Fetchers** (`data_fetchers.py`)
   - **Price Fetcher**: Every 10 seconds for all coins
   - **Volume Fetcher**: Every 2-3 minutes with change detection
   - **Indicator Calculator**: Every 1-3 minutes using cached data
   - **Mystic Fetcher**: Every 1 hour for cosmic data

4. **Strategy System** (`strategy_system.py`)
   - 65+ P-wise strategies per coin
   - Price, volume, technical, momentum, volatility, and cosmic strategies
   - Signal aggregation and confidence scoring

5. **Autobuy System** (`autobuy_system.py`)
   - Automated trade execution
   - Position sizing based on confidence
   - Cooldown management and risk controls

6. **API Layer** (`crypto_autoengine_api.py`)
   - RESTful endpoints for frontend integration
   - `/api/coinstate` for real-time data
   - System control and monitoring endpoints

## 📊 Coin Configuration

### Original 10 Coinbase Coins

- BTC-USD, ETH-USD, ADA-USD, DOT-USD, LINK-USD
- SOL-USD, MATIC-USD, AVAX-USD, ATOM-USD, UNI-USD

### Additional 10 Coinbase Coins

- LTC-USD, BCH-USD, XLM-USD, ALGO-USD, VET-USD
- FIL-USD, ICP-USD, NEAR-USD, FTM-USD, SAND-USD

### Original 10 Binance Coins

- BTCUSDT, ETHUSDT, ADAUSDT, DOTUSDT, LINKUSDT
- SOLUSDT, MATICUSDT, AVAXUSDT, ATOMUSDT, UNIUSDT

### Additional 10 Binance Coins

- LTCUSDT, BCHUSDT, XLMUSDT, ALGOUSDT, VETUSDT
- FILUSDT, ICPUSDT, NEARUSDT, FTMUSDT, SANDUSDT

## 🔄 Data Flow

```mermaid
1. Price Fetcher (10s) → Shared Cache → Price History
2. Volume Fetcher (3m) → Shared Cache → Volume Data
3. Indicator Calculator (2m) → Shared Cache → RSI, MACD, Volatility
4. Mystic Fetcher (1h) → Shared Cache → Cosmic Data
5. Strategy System → 65+ Strategies → Signal Aggregation
6. Autobuy System → Trade Execution → Position Management
7. API Layer → Frontend → Real-time Updates
```

## 🎯 Strategy Categories

### Price-Based Strategies (1-20)

- Price breakouts and breakdowns
- Momentum analysis
- Support/resistance levels
- Trend reversals

### Volume-Based Strategies (21-35)

- Volume spikes and divergence
- Volume trend analysis
- Price-volume relationships

### Technical Indicator Strategies (36-50)

- RSI oversold/overbought
- MACD crossovers and divergence
- Moving average analysis

### Momentum Strategies (51-60)

- Short-term momentum (1m, 5m, 15m, 1h)
- Momentum convergence
- Multi-timeframe analysis

### Volatility Strategies (61-65)

- High volatility breakouts
- Low volatility consolidation
- Volatility-based entries

### Cosmic/Mystic Strategies (66+)

- Solar flare impact
- Schumann resonance
- Lunar phase influence
- Cosmic alignment factors

## 🚀 API Endpoints

### Core Endpoints

#### `/api/coinstate`

- **GET**: Get all coin states for frontend
- **Refresh**: Every 5-10 seconds
- **Data**: Price, volume, indicators, strategies, cosmic data

#### `/api/coinstate/{symbol}`

- **GET**: Get state for specific coin
- **Parameters**: Symbol (e.g., BTC-USD, BTCUSDT)

#### `/api/coins`

- **GET**: Get all configured coins
- **Returns**: All symbols, enabled coins, exchange breakdowns

### Strategy Endpoints

#### `/api/strategies`

- **GET**: Get all strategy signals
- **Returns**: 65+ strategies per coin

#### `/api/strategies/{symbol}`

- **GET**: Get strategies for specific coin

### Trading Endpoints

#### `/api/trading`

- **GET**: Get trading information and statistics

#### `/api/trading/start`

- **POST**: Start autobuy system

#### `/api/trading/stop`

- **POST**: Stop autobuy system

#### `/api/trading/cancel/{symbol}`

- **POST**: Cancel pending order for specific coin

#### `/api/trading/cancel-all`

- **POST**: Cancel all pending orders

### System Control

#### `/api/system/status`

- **GET**: Get overall system status

#### `/api/system/start`

- **POST**: Start all system components

#### `/api/system/stop`

- **POST**: Stop all system components

#### `/api/health`

- **GET**: Health check endpoint

## 📊 Monitoring & Logging

### Log Files

- `crypto_autoengine.log`: Main application log
- Redis logs: If Redis is enabled

### Key Metrics

- **Total API calls/minute**: ~140 calls
- **Strategy execution time**: <1 second per coin
- **Cache hit rate**: >95%
- **Trade execution rate**: Based on signal strength

### Health Monitoring

```bash
# Check system health
curl http://localhost:8000/api/health

# Get system status
curl http://localhost:8000/api/system/status
```

## 🔧 Configuration

### Fetcher Intervals

```python
price_fetch_interval: int = 10      # seconds
volume_fetch_interval: int = 180    # 3 minutes
indicator_calc_interval: int = 120  # 2 minutes
mystic_fetch_interval: int = 3600   # 1 hour
```

### Strategy Thresholds

```python
min_confidence: float = 0.7
max_confidence: float = 0.95
min_signal_strength: float = 0.6
cooldown_period: int = 300          # 5 minutes
```

### Trading Limits

```python
min_trade_amount: float = 10.0
max_trade_amount: float = 10000.0   # For BTC/ETH
max_trade_amount: float = 5000.0    # For other coins
```

## 🚨 Risk Management

### Built-in Protections

- **Cooldown periods**: 5 minutes between trades per coin
- **Confidence thresholds**: Minimum 70% confidence required
- **Signal strength**: Minimum 60% strength required
- **Position sizing**: Scaled by confidence level
- **Trade limits**: Per-coin maximum amounts

### Monitoring

- Real-time trade tracking
- Success/failure statistics
- Profit/loss calculation
- Order cancellation capabilities

## 🔮 Future Enhancements

### Planned Features

- **Machine Learning Integration**: Enhanced signal prediction
- **Portfolio Management**: Multi-coin position balancing
- **Risk Analytics**: Advanced risk assessment
- **Backtesting Engine**: Strategy performance validation
- **Mobile App**: Real-time trading notifications

### API Integrations

- **Additional Exchanges**: Kraken, Gemini, etc.
- **News APIs**: Sentiment analysis integration
- **Social Media**: Twitter/Reddit sentiment
- **On-chain Data**: Blockchain analytics

## 📞 Support

### Documentation

- API documentation: `http://localhost:8000/docs`
- Interactive API explorer: `http://localhost:8000/redoc`

### Logging

- Application logs: `crypto_autoengine.log`
- Error tracking: Check log files for detailed error messages

### Performance Tuning

- Adjust fetch intervals based on API limits
- Modify strategy thresholds for different market conditions
- Fine-tune cache TTL for optimal performance

---

**CRYPTO AUTOENGINE** - Complete cryptocurrency trading automation with cosmic intelligence.
