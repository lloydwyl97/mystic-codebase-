# 🤖 AI Strategy Execution System - Complete Guide

## 🌐 Windows 11 Home + Docker Deployment

This guide covers the complete AI strategy execution system that automatically executes trades based on AI signals, leaderboard rankings, and advanced trading algorithms.

## 📋 System Components

### 1. **AI Strategy Execution** (`ai_strategy_execution.py`)
- Executes trades based on AI signals
- Supports Binance.us and Coinbase
- Real-time price monitoring
- Automatic exchange selection (best price)
- Discord/Telegram notifications

### 2. **AI Leaderboard Executor** (`ai_leaderboard_executor.py`)
- Monitors strategy leaderboard
- Executes top-performing strategies
- Configurable win rate and profit thresholds
- Continuous execution every hour
- Strategy ranking and selection

### 3. **AI Trade Engine** (`ai_trade_engine.py`)
- Advanced trading with take-profit and trailing-stop
- Auto-buy signals with position management
- Real-time dashboard output
- Risk management features
- Continuous monitoring and execution

### 4. **Mutation Leaderboard** (`mutation_leaderboard.json`)
- Strategy performance tracking
- Win rate and profit metrics
- Auto-updated by AI mutation engine
- Strategy ranking system

## 🚀 Quick Start

### 1. Set Up Environment

```powershell
# Run as Administrator (recommended)
# Right-click PowerShell and select "Run as Administrator"

# Navigate to your Mystic-Codebase directory
cd C:\path\to\Mystic-Codebase

# Set up the environment (if not already done)
.\setup-env.ps1
```

### 2. Configure AI Strategy Settings

Edit `backend/.env` and add your configuration:

```env
# ===== AI STRATEGY CONFIGURATION =====
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET_KEY=your_coinbase_secret_key
COINBASE_PASSPHRASE=your_coinbase_passphrase

# ===== TRADING SETTINGS =====
SYMBOL_PAIR_BINANCE=ETHUSDT
SYMBOL_PAIR_COINBASE=ETH-USD
USD_TRADE_AMOUNT=50
TAKE_PROFIT_PERCENTAGE=0.15
TRAILING_STOP_PERCENTAGE=0.03

# ===== LEADERBOARD SETTINGS =====
MIN_WIN_RATE=0.55
MIN_PROFIT=10.0

# ===== NOTIFICATIONS =====
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
DISCORD_WEBHOOK=your_discord_webhook_url
```

### 3. Deploy AI Strategy System

```powershell
# Deploy all AI strategy components
.\run-ai-strategies.ps1
```

## 🔧 Configuration Options

### Trading Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `SYMBOL_PAIR_BINANCE` | Binance trading pair | ETHUSDT |
| `SYMBOL_PAIR_COINBASE` | Coinbase trading pair | ETH-USD |
| `USD_TRADE_AMOUNT` | Trade amount in USD | 50 |
| `TAKE_PROFIT_PERCENTAGE` | Take profit percentage | 0.15 (15%) |
| `TRAILING_STOP_PERCENTAGE` | Trailing stop percentage | 0.03 (3%) |

### Leaderboard Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `MIN_WIN_RATE` | Minimum win rate for strategy selection | 0.55 (55%) |
| `MIN_PROFIT` | Minimum profit for strategy selection | 10.0 |

### API Key Requirements

#### Binance API Keys
- **Permissions needed**: Read Info, Spot & Margin Trading
- **Get from**: https://www.binance.us/en/my/settings/api-management
- **Security**: Enable IP restrictions

#### Coinbase API Keys
- **Permissions needed**: View, Transfer
- **Get from**: https://pro.coinbase.us/profile/api
- **Security**: Enable IP restrictions

## 🐳 Docker Deployment

### Automatic Deployment
```powershell
# One-command deployment
.\run-ai-strategies.ps1
```

### Manual Docker Commands
```powershell
# Build AI Strategy Execution
cd backend
docker build -f ai_strategy_execution.Dockerfile -t mystic-ai-strategy .

# Build AI Leaderboard Executor
docker build -f ai_leaderboard_executor.Dockerfile -t mystic-ai-leaderboard .

# Build AI Trade Engine
docker build -f ai_trade_engine.Dockerfile -t mystic-ai-trade-engine .

# Run containers
docker run -d --name mystic-ai-strategy --env-file .env -v "${PWD}/logs:/app/logs" mystic-ai-strategy
docker run -d --name mystic-ai-leaderboard --env-file .env -v "${PWD}/logs:/app/logs" -v "${PWD}/mutation_leaderboard.json:/app/mutation_leaderboard.json" mystic-ai-leaderboard
docker run -d --name mystic-ai-trade-engine --env-file .env -v "${PWD}/logs:/app/logs" mystic-ai-trade-engine
```

### Container Management
```powershell
# View logs
docker logs -f mystic-ai-strategy
docker logs -f mystic-ai-leaderboard
docker logs -f mystic-ai-trade-engine

# Stop all components
docker stop mystic-ai-strategy mystic-ai-leaderboard mystic-ai-trade-engine

# Restart all components
docker restart mystic-ai-strategy mystic-ai-leaderboard mystic-ai-trade-engine
```

## 📊 Monitoring & Logs

### Log Files
- **AI Strategy Execution**: `backend/logs/ai_strategy_execution.log`
- **AI Leaderboard Executor**: `backend/logs/ai_leaderboard_executor.log`
- **AI Trade Engine**: `backend/logs/ai_trade_engine.log`
- **Leaderboard Data**: `backend/mutation_leaderboard.json`

### Sample Log Output
```
2024-01-15 12:00:00 - ai_strategy_execution - INFO - AI Strategy Execution System - Connection Test
2024-01-15 12:01:00 - ai_leaderboard_executor - INFO - Selected strategy: strategy_47b_rsi_breakout (win_rate: 0.64, profit: 22.3)
2024-01-15 12:01:00 - ai_trade_engine - INFO - Position opened: Binance ETHUSDT @ $1824.55
```

### Dashboard Output
```
📊 Position: binance | Entry: $1824.55 | Current: $1830.20 | PnL: 0.31% | Peak: $1830.20
📊 No Position | Binance: $1824.55 | Coinbase: $1825.10
```

## 🔍 Troubleshooting

### Common Issues

#### API Key Errors
1. **Verify API keys** in `backend/.env`
2. **Check permissions** - ensure trading is enabled
3. **Test API access** manually first
4. **Check IP restrictions** if enabled

#### Strategy Execution Failures
1. **Check leaderboard file** exists and is valid JSON
2. **Verify strategy criteria** (win rate, profit thresholds)
3. **Review logs** for specific errors
4. **Check exchange connectivity**

#### Trade Engine Issues
1. **Verify take-profit and trailing-stop** settings
2. **Check position management** logic
3. **Review price feed** connectivity
4. **Monitor notification** delivery

### Performance Optimization

#### For High-Frequency Trading
```env
# Reduce check intervals
CHECK_INTERVAL=30  # 30 seconds
```

#### For Conservative Trading
```env
# Increase thresholds
MIN_WIN_RATE=0.65
MIN_PROFIT=15.0
TAKE_PROFIT_PERCENTAGE=0.20
TRAILING_STOP_PERCENTAGE=0.05
```

## 🔒 Security Best Practices

### API Key Security
1. **Use dedicated API keys** for AI strategy execution
2. **Enable IP restrictions** to your server IP
3. **Disable futures trading** if not needed
4. **Regularly rotate API keys**
5. **Monitor API usage** for suspicious activity

### Trading Security
1. **Start with small amounts** for testing
2. **Monitor all trades** in real-time
3. **Set reasonable limits** on trade sizes
4. **Use stop-losses** and take-profits
5. **Keep detailed logs** of all activities

## 📱 Notifications

### Discord Notifications
```
✅ Binance Buy Executed: ETHUSDT | $50
{"orderId": 123456789, "status": "FILLED"}

🎯 Take Profit Hit
🛑 Trailing Stop Hit
```

### Telegram Notifications
```
✅ Binance Buy Executed: ETHUSDT | $50
{"orderId": 123456789, "status": "FILLED"}

🎯 Take Profit Hit
🛑 Trailing Stop Hit
```

## 🚀 Advanced Features

### Multiple Strategy Execution
The system can run multiple strategies simultaneously:

```powershell
# Run multiple instances for different strategies
docker run -d --name mystic-ai-strategy-1 --env-file backend/.env.strategy1 mystic-ai-strategy
docker run -d --name mystic-ai-strategy-2 --env-file backend/.env.strategy2 mystic-ai-strategy
```

### Custom Strategy Integration
You can integrate custom strategies by modifying the signal generation:

```python
def check_for_buy_signal():
    # Add your custom signal logic here
    # Integrate with your existing AI systems
    return your_custom_signal_logic()
```

### Integration with Main Platform
The AI strategy system integrates with your main Mystic Trading Platform:

- **Shared configuration** via `.env` file
- **Unified logging** in `backend/logs/`
- **Dashboard integration** for monitoring
- **API endpoints** for status checking

## 📞 Support

If you encounter issues:

1. **Check the logs**: `docker logs mystic-ai-strategy`
2. **Verify configuration** in `backend/.env`
3. **Test API keys** manually
4. **Review this documentation**
5. **Check exchange status** pages

## ✨ Features Summary

- ✅ **AI Strategy Execution** with multi-exchange support
- ✅ **Leaderboard-based Strategy Selection** with performance metrics
- ✅ **Advanced Trade Engine** with take-profit and trailing-stop
- ✅ **Real-time Notifications** via Discord/Telegram
- ✅ **Comprehensive Logging** and monitoring
- ✅ **Docker Deployment** for reliability
- ✅ **Windows 11 Home optimized**
- ✅ **Security best practices**
- ✅ **Easy setup scripts**

---

**🤖 Your AI Strategy Execution System is now ready for automated trading! 🚀**
