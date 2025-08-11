# Mystic Trading Bots - Separate Exchange Bots

This document describes the new separate trading bot system that runs independent bots for Coinbase and Binance exchanges.

## Overview

The trading system now consists of three main components:

1. **Coinbase Bot** (`coinbase_bot.py`) - Handles only Coinbase coins
2. **Binance Bot** (`binance_bot.py`) - Handles only Binance coins  
3. **Bot Manager** (`bot_manager.py`) - Coordinates both bots

## Bot Configuration

### Coinbase Bot

- **Coins**: AR, OP, IMX, ODEAN, MINA, ANKR, GRT, MATIC, RNDR, LINK
- **Rate Limit**: 10 requests per minute (1 request every 6 seconds per coin)
- **API**: Coinbase Pro API
- **Features**:
  - Real-time price fetching
  - Signal generation
  - Trade execution (simulated)
  - Error handling and auto-restart

### Binance Bot

- **Coins**: AKT, KUJI, OSMO, STRD, EVMOS, SCRT, STARS, DYDX, SOMM, CRE
- **Rate Limit**: 30 requests per minute (1 request every 2-4 seconds per coin)
- **API**: Binance API v3
- **Features**:
  - Real-time price fetching with 24h change data
  - Enhanced signal generation with technical analysis
  - Trade execution (simulated)
  - Error handling and auto-restart

## Rate Limiting

### Coinbase Bot Rate Limits

- **Per-minute limit**: 10 requests
- **Per-coin interval**: 6 seconds minimum
- **Throttling**: Automatic rate limit checking and waiting

### Binance Bot Rate Limits

- **Per-minute limit**: 30 requests  
- **Per-coin interval**: 2-4 seconds (randomized)
- **Throttling**: Automatic rate limit checking and waiting

## Error Handling

Both bots include comprehensive error handling:

- **Network errors**: Automatic retry with exponential backoff
- **API errors**: Graceful degradation and fallback
- **Rate limit violations**: Automatic throttling and waiting
- **Bot crashes**: Auto-restart with configurable attempts
- **Resource cleanup**: Proper session and connection cleanup

## API Endpoints

### Bot Management

- `POST /api/bot/start` - Start bot manager (both bots)
- `POST /api/bot/stop` - Stop bot manager
- `GET /api/bot/status` - Get overall bot status

### Individual Bot Control

- `POST /api/bot/coinbase/start` - Start Coinbase bot only
- `POST /api/bot/coinbase/stop` - Stop Coinbase bot only
- `GET /api/bot/coinbase/status` - Get Coinbase bot status
- `GET /api/bot/coinbase/data` - Get Coinbase market data

- `POST /api/bot/binance/start` - Start Binance bot only
- `POST /api/bot/binance/stop` - Stop Binance bot only
- `GET /api/bot/binance/status` - Get Binance bot status
- `GET /api/bot/binance/data` - Get Binance market data

## Running the Bots

### Option 1: Run Both Bots Together

```bash
# From project root
python backend/run_bots.py
```

### Option 2: Run Individual Bots

```bash
# Coinbase bot only
python backend/coinbase_bot.py

# Binance bot only  
python backend/binance_bot.py
```

### Option 3: Windows Batch File

```bash
# Double-click or run from command line
run_trading_bots.bat
```

## Logging

Each bot creates its own log file:

- `coinbase_bot.log` - Coinbase bot logs
- `binance_bot.log` - Binance bot logs
- `bot_manager.log` - Bot manager logs
- `trading_bots.log` - Combined logs when using run_bots.py

## Monitoring

### Bot Status Response

```json
{
  "bot_type": "coinbase",
  "status": {
    "status": "running",
    "is_running": true,
    "coins_tracked": 10,
    "market_data_count": 10,
    "total_trades": 5,
    "successful_trades": 4,
    "failed_trades": 1,
    "last_update": "2024-01-15T10:30:00Z",
    "rate_limit": {
      "requests_this_minute": 8,
      "max_requests_per_minute": 10
    }
  }
}
```

### Market Data Response

```json
{
  "bot_type": "coinbase",
  "market_data": {
    "AR": {
      "symbol": "AR",
      "price": 45.23,
      "change_24h": 2.5,
      "volume_24h": 1234567,
      "timestamp": "2024-01-15T10:30:00Z",
      "api_source": "coinbase"
    }
  },
  "signals": [
    {
      "symbol": "AR",
      "action": "BUY",
      "confidence": 75.0,
      "price": 45.23,
      "timestamp": "2024-01-15T10:30:00Z",
      "reason": "Strong upward momentum (+2.50%)"
    }
  ]
}
```

## Configuration

### Trading Configuration

Both bots support configurable trading parameters:

- `enabled`: Enable/disable trading

- `max_investment`: Maximum investment per trade
- `stop_loss`: Stop loss percentage
- `take_profit`: Take profit percentage
- `min_confidence`: Minimum signal confidence for trades

### Rate Limiting Configuration

Rate limits are hardcoded for safety but can be modified in the bot files:

- Coinbase: 10 requests/minute, 6s interval
- Binance: 30 requests/minute, 2-4s interval

## Safety Features

1. **Rate Limiting**: Prevents API abuse and account suspension
2. **Error Recovery**: Automatic restart on failures
3. **Resource Management**: Proper cleanup of connections and sessions
4. **Logging**: Comprehensive logging for debugging and monitoring
5. **Graceful Shutdown**: Proper cleanup on exit
6. **Signal Handling**: Responds to Ctrl+C and system signals

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**: Bots automatically handle rate limits, but check logs for frequency
2. **Network Errors**: Check internet connection and API availability
3. **Import Errors**: Ensure all dependencies are installed
4. **Permission Errors**: Check file permissions for log files

### Log Analysis

```bash
# Check Coinbase bot logs
tail -f backend/coinbase_bot.log

# Check Binance bot logs  
tail -f backend/binance_bot.log

# Check for errors
grep "ERROR" backend/*.log
```

## Performance

### Expected Performance

- **Coinbase Bot**: ~10 API calls/minute, 6-second intervals
- **Binance Bot**: ~30 API calls/minute, 2-4 second intervals
- **Memory Usage**: ~50-100MB per bot
- **CPU Usage**: Minimal (mostly I/O bound)

### Optimization Tips

1. Monitor rate limit usage in logs
2. Adjust coin lists to focus on high-priority coins
3. Use caching to reduce API calls
4. Monitor bot health via status endpoints

## Future Enhancements

1. **Database Integration**: Store trade history and performance metrics
2. **Web Interface**: Real-time bot monitoring dashboard
3. **Advanced Strategies**: Implement more sophisticated trading algorithms
4. **Multi-Exchange Support**: Add support for additional exchanges
5. **Backtesting**: Historical strategy testing capabilities
6. **Risk Management**: Advanced position sizing and risk controls
