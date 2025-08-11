#!/usr/bin/env python3
"""
Binance US Autobuy Setup Script
Setup and configuration for SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT autobuy system
"""

import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutobuySetup:
    """Setup and configuration for the autobuy system"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.config_file = self.project_root / "autobuy_config.json"

    def create_directories(self):
        """Create necessary directories"""
        directories = ["logs", "reports", "data", "backups"]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")

    def create_env_template(self):
        """Create .env template file"""
        env_template = """# Binance US Autobuy System Configuration
# ================================================

# Binance US API Configuration
BINANCE_US_API_KEY=your_binance_us_api_key_here
BINANCE_US_SECRET_KEY=your_binance_us_secret_key_here

# Trading Configuration
TRADING_ENABLED=true
BINANCE_TESTNET=false
USD_AMOUNT_PER_TRADE=50
MAX_CONCURRENT_TRADES=4

# Signal Configuration
MIN_VOLUME_INCREASE=1.5
MIN_PRICE_CHANGE=0.02
SIGNAL_COOLDOWN=300

# Risk Management
MAX_DAILY_TRADES=48
MAX_DAILY_VOLUME=2000.0
STOP_LOSS_PERCENTAGE=0.05
TAKE_PROFIT_PERCENTAGE=0.10

# Notification Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here

# Logging Configuration
LOG_LEVEL=INFO

# Performance Configuration
CYCLE_INTERVAL=30
DATA_CACHE_TTL=60

# Advanced Configuration
ENABLE_TECHNICAL_ANALYSIS=true
ENABLE_SENTIMENT_ANALYSIS=false
ENABLE_WHALE_TRACKING=true

# Market Hours (timezone.utc)
TRADING_START_HOUR=0
TRADING_END_HOUR=24

# Emergency Configuration
EMERGENCY_STOP=false
MAX_LOSS_PER_TRADE=10.0
"""

        if not self.env_file.exists():
            with open(self.env_file, "w") as f:
                f.write(env_template)
            logger.info("✅ Created .env template file")
            logger.info("⚠️  Please configure your API keys in .env file")
        else:
            logger.info("ℹ️  .env file already exists")

    def create_config_file(self):
        """Create configuration file"""
        config = {
            "trading_pairs": {
                "SOLUSDT": {
                    "name": "Solana",
                    "min_trade_amount": 25.0,
                    "max_trade_amount": 200.0,
                    "target_frequency": 15,
                    "enabled": True,
                },
                "BTCUSDT": {
                    "name": "Bitcoin",
                    "min_trade_amount": 50.0,
                    "max_trade_amount": 500.0,
                    "target_frequency": 30,
                    "enabled": True,
                },
                "ETHUSDT": {
                    "name": "Ethereum",
                    "min_trade_amount": 50.0,
                    "max_trade_amount": 400.0,
                    "target_frequency": 20,
                    "enabled": True,
                },
                "AVAXUSDT": {
                    "name": "Avalanche",
                    "min_trade_amount": 25.0,
                    "max_trade_amount": 200.0,
                    "target_frequency": 15,
                    "enabled": True,
                },
            },
            "signal_config": {
                "min_confidence": 50.0,
                "min_volume_increase": 1.5,
                "min_price_change": 0.02,
                "max_price_change": 0.15,
                "volume_threshold": 1000000,
                "volatility_threshold": 0.05,
                "momentum_threshold": 0.03,
            },
            "risk_config": {
                "max_concurrent_trades": 4,
                "max_daily_trades": 48,
                "max_daily_volume": 2000.0,
                "stop_loss_percentage": 0.05,
                "take_profit_percentage": 0.10,
                "max_drawdown": 0.20,
            },
            "trading_hours": {
                "start_hour": 0,
                "end_hour": 24,
                "timezone": "timezone.utc",
            },
            "emergency_stop": False,
            "cycle_interval": 30,
            "signal_cooldown": 300,
        }

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("✅ Created configuration file")

    def create_requirements_file(self):
        """Create requirements.txt file"""
        requirements = """# Binance US Autobuy System Requirements
# ================================================

# Core dependencies
aiohttp>=3.8.0
requests>=2.28.0
python-dotenv>=0.19.0
fastapi>=0.68.0
uvicorn>=0.15.0
websockets>=10.0

# Data processing
pandas>=1.5.0
numpy>=1.21.0

# Logging and monitoring
structlog>=21.5.0

# Optional: Advanced features
# ccxt>=2.0.0  # For additional exchange support
# ta>=0.10.0   # Removed - using custom implementations
# openai>=0.27.0  # For AI features
"""

        requirements_file = self.project_root / "requirements_autobuy.txt"
        with open(requirements_file, "w") as f:
            f.write(requirements)

        logger.info("✅ Created requirements_autobuy.txt")

    def create_startup_scripts(self):
        """Create startup scripts"""
        # Windows batch script
        windows_script = """@echo off
echo Starting Binance US Autobuy System...
echo.
echo Trading Pairs: SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT
echo Dashboard: http://localhost:8080
echo.
python launch_autobuy.py
pause
"""

        windows_file = self.project_root / "start_autobuy.bat"
        with open(windows_file, "w") as f:
            f.write(windows_script)

        # PowerShell script
        powershell_script = """# Binance US Autobuy System Startup Script
Write-Host "Starting Binance US Autobuy System..." -ForegroundColor Green
Write-Host ""
Write-Host "Trading Pairs: SOLUSDT, BTCUSDT, ETHUSDT, AVAXUSDT" -ForegroundColor Yellow
Write-Host "Dashboard: http://localhost:8080" -ForegroundColor Yellow
Write-Host ""

try {
    python launch_autobuy.py
} catch {
    Write-Host "Error starting system: $_" -ForegroundColor Red
}

Read-Host "Press Enter to exit"
"""

        powershell_file = self.project_root / "start_autobuy.ps1"
        with open(powershell_file, "w") as f:
            f.write(powershell_script)

        logger.info("✅ Created startup scripts")

    def create_readme(self):
        """Create README file"""
        readme = """# Binance US Autobuy System

## Overview
Automated trading system for Binance US focusing on SOLUSDT, BTCUSDT, ETHUSDT, and AVAXUSDT with aggressive autobuy strategy.

## Features
- 🎯 Focused on 4 major trading pairs
- 💰 Aggressive autobuy strategy
- 📊 Real-time dashboard monitoring
- 🔔 Telegram/Discord notifications
- 🛡️ Risk management controls
- 📈 Performance reporting

## Quick Start

### 1. Setup Environment
```bash
python setup_autobuy.py
```

### 2. Configure API Keys
Edit `.env` file and add your Binance US API credentials:
```
BINANCE_US_API_KEY=your_api_key_here
BINANCE_US_SECRET_KEY=your_secret_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements_autobuy.txt
```

### 4. Start System
```bash
# Windows
start_autobuy.bat

# PowerShell
./start_autobuy.ps1

# Python
python launch_autobuy.py
```

### 5. Access Dashboard
Open http://localhost:8080 in your browser

## Configuration

### Trading Pairs
- **SOLUSDT**: Solana - $25-$200 per trade, 15min frequency
- **BTCUSDT**: Bitcoin - $50-$500 per trade, 30min frequency
- **ETHUSDT**: Ethereum - $50-$400 per trade, 20min frequency
- **AVAXUSDT**: Avalanche - $25-$200 per trade, 15min frequency

### Signal Parameters
- Minimum confidence: 50%
- Volume increase: 50% above average
- Price change: 2% minimum
- Signal cooldown: 5 minutes

### Risk Management
- Max concurrent trades: 4
- Max daily trades: 48
- Max daily volume: $2000
- Stop loss: 5%
- Take profit: 10%

## Safety Features
- ⚠️ Emergency stop capability
- 🛡️ Maximum loss per trade limits
- 📊 Real-time monitoring
- 🔔 Instant notifications
- 📈 Performance tracking

## Files Structure
```
backend/
├── binance_us_autobuy.py      # Main autobuy engine
├── autobuy_config.py          # Configuration management
├── autobuy_dashboard.py       # Web dashboard
├── autobuy_report.py          # Reporting system
├── launch_autobuy.py          # System launcher
├── setup_autobuy.py           # Setup script
├── .env                       # Environment variables
├── logs/                      # Log files
├── reports/                   # Performance reports
└── data/                      # Data storage
```

## Monitoring
- **Dashboard**: http://localhost:8080
- **Logs**: `logs/autobuy_launcher.log`
- **Reports**: `reports/autobuy_report_*.json`

## Notifications
Configure Telegram and Discord notifications in `.env`:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
DISCORD_WEBHOOK_URL=your_webhook_url
```

## ⚠️ WARNING
This system executes real trades with real money. Ensure you:
- Understand the risks involved
- Start with small amounts
- Monitor the system closely
- Have proper API permissions
- Test thoroughly before live trading

## Support
For issues and questions, check the logs in the `logs/` directory.

## License
This software is for educational purposes. Use at your own risk.
"""

        readme_file = self.project_root / "README_AUTOBUY.md"
        with open(readme_file, "w") as f:
            f.write(readme)

        logger.info("✅ Created README_AUTOBUY.md")

    def validate_setup(self) -> bool:
        """Validate the setup"""
        required_files = [
            ".env",
            "autobuy_config.json",
            "requirements_autobuy.txt",
            "start_autobuy.bat",
            "start_autobuy.ps1",
            "README_AUTOBUY.md",
        ]

        required_dirs = ["logs", "reports", "data", "backups"]

        missing_files = []
        for file in required_files:
            if not (self.project_root / file).exists():
                missing_files.append(file)

        missing_dirs = []
        for directory in required_dirs:
            if not (self.project_root / directory).exists():
                missing_dirs.append(directory)

        if missing_files or missing_dirs:
            logger.error("❌ Setup validation failed:")
            if missing_files:
                logger.error(f"   Missing files: {', '.join(missing_files)}")
            if missing_dirs:
                logger.error(f"   Missing directories: {', '.join(missing_dirs)}")
            return False

        logger.info("✅ Setup validation passed")
        return True

    def run_setup(self):
        """Run complete setup"""
        logger.info("🚀 Starting Binance US Autobuy System Setup...")
        logger.info("=" * 60)

        try:
            # Create directories
            self.create_directories()

            # Create configuration files
            self.create_env_template()
            self.create_config_file()
            self.create_requirements_file()

            # Create startup scripts
            self.create_startup_scripts()

            # Create documentation
            self.create_readme()

            # Validate setup
            if self.validate_setup():
                logger.info("✅ Setup completed successfully!")
                logger.info("=" * 60)
                logger.info("📋 Next steps:")
                logger.info("1. Configure your API keys in .env file")
                logger.info("2. Install dependencies: pip install -r requirements_autobuy.txt")
                logger.info("3. Start the system: python launch_autobuy.py")
                logger.info("4. Access dashboard: http://localhost:8080")
                logger.info("=" * 60)
            else:
                logger.error("❌ Setup validation failed")
                return False

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

        return True


def main():
    """Main setup function"""
    setup = AutobuySetup()

    if setup.run_setup():
        print("\n🎉 Setup completed successfully!")
        print("📖 Check README_AUTOBUY.md for detailed instructions")
        return 0
    else:
        print("\n❌ Setup failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
