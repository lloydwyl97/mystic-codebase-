# 🔐 Mystic Auto-Withdraw System - Complete Guide

## 🌐 Windows 11 Home + Docker Deployment

This guide will help you deploy the Mystic Auto-Withdraw system that automatically transfers funds to your cold wallet when thresholds are reached.

## 📋 What It Does

The auto-withdraw system:
- **Monitors your exchange balance** continuously
- **Automatically withdraws** when balance exceeds threshold
- **Supports Binance and Coinbase** exchanges
- **Sends notifications** via Discord/Telegram
- **Logs all activities** for audit trail
- **Runs in Docker** for reliability

## 🚀 Quick Start

### 1. Set Up Environment

```powershell
# Run as Administrator (recommended)
# Right-click PowerShell and select "Run as Administrator"

# Navigate to your Mystic-Codebase directory
cd C:\path\to\Mystic-Codebase

# Set up the auto-withdraw system
.\setup-auto-withdraw.ps1
```

### 2. Configure Your Settings

Edit `backend/.env` and add your configuration:

```env
# ===== AUTO-WITHDRAW CONFIGURATION =====
EXCHANGE=binance                    # or coinbase
COLD_WALLET_ADDRESS=0xYourExodusWalletHere
COLD_WALLET_THRESHOLD=250.00
CHECK_INTERVAL=60                   # seconds

# ===== EXCHANGE API KEYS =====
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

COINBASE_API_KEY=your_coinbase_api_key
COINBASE_API_SECRET=your_coinbase_api_secret
COINBASE_PASSPHRASE=your_coinbase_passphrase

# ===== NOTIFICATIONS =====
DISCORD_WEBHOOK=your_discord_webhook_url
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 3. Deploy with Docker

```powershell
# Deploy the auto-withdraw system
.\run-auto-withdraw.ps1
```

## 🔧 Configuration Options

### Exchange Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `EXCHANGE` | Exchange to monitor (binance/coinbase) | binance |
| `COLD_WALLET_ADDRESS` | Your cold wallet address | Required |
| `COLD_WALLET_THRESHOLD` | Minimum balance to maintain | 250.00 |
| `CHECK_INTERVAL` | How often to check balance (seconds) | 60 |

### API Key Requirements

#### Binance API Keys
- **Permissions needed**: Read Info, Spot & Margin Trading, Withdraw
- **Get from**: https://www.binance.us/en/my/settings/api-management
- **Security**: Enable IP restrictions, disable futures trading

#### Coinbase API Keys
- **Permissions needed**: View, Transfer
- **Get from**: https://pro.coinbase.us/profile/api
- **Security**: Enable IP restrictions

### Notification Settings

#### Discord Webhook
1. Go to your Discord server settings
2. Navigate to Integrations → Webhooks
3. Create a new webhook
4. Copy the webhook URL to `DISCORD_WEBHOOK`

#### Telegram Bot
1. Message @BotFather on Telegram
2. Create a new bot with `/newbot`
3. Get your bot token
4. Start a chat with your bot
5. Get your chat ID from @userinfobot
6. Add `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`

## 🐳 Docker Deployment

### Automatic Deployment
```powershell
# One-command deployment
.\run-auto-withdraw.ps1
```

### Manual Docker Commands
```powershell
# Build the image
cd backend
docker build -f auto_withdraw.Dockerfile -t mystic-auto-withdraw .

# Run the container
docker run -d \
    --name mystic-auto-withdraw \
    --env-file .env \
    -v "${PWD}/logs:/app/logs" \
    --restart unless-stopped \
    mystic-auto-withdraw
```

### Container Management
```powershell
# View logs
docker logs -f mystic-auto-withdraw

# Stop the system
docker stop mystic-auto-withdraw

# Restart the system
docker restart mystic-auto-withdraw

# Remove the container
docker rm mystic-auto-withdraw
```

## 📊 Monitoring & Logs

### Log Files
- **Application logs**: `backend/logs/auto_withdraw.log`
- **Withdrawal history**: `backend/logs/withdrawals.json`
- **Docker logs**: `docker logs mystic-auto-withdraw`

### Sample Log Output
```
2024-01-15 12:00:00 - auto_withdraw - INFO - Auto-withdraw system initialized for binance
2024-01-15 12:00:00 - auto_withdraw - INFO - Cold wallet threshold: $250.00
2024-01-15 12:01:00 - auto_withdraw - INFO - [BINANCE] Current USDT balance: $500.25
2024-01-15 12:01:00 - auto_withdraw - INFO - [BINANCE] Withdrawal successful: $250.25
```

### Sample Withdrawal Log
```json
{
  "timestamp": "2024-01-15T12:01:00.123456",
  "exchange": "binance",
  "amount": 250.25,
  "status": "success",
  "details": {
    "id": "123456789",
    "status": "SUCCESS"
  },
  "cold_wallet_address": "0x1234..."
}
```

## 🔍 Troubleshooting

### Common Issues

#### Docker Desktop Not Running
```powershell
# Start Docker Desktop manually
# Or check if it's running
docker version
```

#### API Key Errors
1. **Verify API keys** in `backend/.env`
2. **Check permissions** - ensure withdrawal is enabled
3. **Test API access** manually first
4. **Check IP restrictions** if enabled

#### Balance Not Updating
1. **Check exchange API** status
2. **Verify network connectivity**
3. **Check rate limits** - increase `CHECK_INTERVAL`
4. **Review logs** for specific errors

#### Withdrawal Failures
1. **Verify cold wallet address** is correct
2. **Check network selection** (ETH, BSC, etc.)
3. **Ensure sufficient balance** for fees
4. **Review exchange withdrawal limits**

### Performance Optimization

#### For High-Frequency Monitoring
```env
CHECK_INTERVAL=30  # Check every 30 seconds
```