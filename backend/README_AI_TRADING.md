# Mystic AI Trading Platform

A complete AI-powered trading system with mystic data integration, self-learning capabilities, and comprehensive monitoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Windows 11 Home (tested)

### Installation

1. **Clone and setup:**
```bash
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Environment Variables (optional):**
Create `.env` file:
```env
DISCORD_WEBHOOK=your_discord_webhook_url
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

3. **Start with Docker:**
```bash
docker-compose up -d
```

4. **Start manually:**
```bash
python start_ai_trading.py
```

## 🧠 AI Features

### Core Components
- **AI Mode Controller**: Switch between training/live/off modes
- **Auto Learner**: Self-adjusting strategy parameters
- **Simulation Logger**: Track all trade decisions
- **Performance Monitor**: Real-time performance tracking
- **Mystic Integration**: Tesla 369, Faerie Star, Lagos alignment

### AI Modes
- **Training Mode**: Learn without real trades
- **Live Mode**: Execute real trades with safety limits
- **Off Mode**: Disable AI completely

### Safety Features
- Daily trade limits
- Maximum drawdown protection
- Automatic stagnation detection
- Model rollback capabilities

## 📊 Dashboard & Monitoring

### API Endpoints
- `GET /ai/dashboard` - Complete AI status
- `GET /ai/performance` - Performance metrics
- `GET /ai/rating` - AI self-assessment
- `GET /ai/trades/recent` - Recent trades
- `POST /ai/mode/{mode}` - Change AI mode

### Notifications
- Discord integration
- Telegram alerts
- Daily performance summaries
- Critical alerts for issues

## 📈 Performance Tracking

### Metrics Tracked
- Total trades and profit
- Win/loss ratios
- Average profit per trade
- AI confidence levels
- Strategy adjustments

### Charts & Reports
- Trade visualization
- Performance over time
- Strategy analysis
- Export to JSON/CSV

## 🔧 Configuration

### AI Parameters
- Confidence threshold (0.5-0.95)
- Daily trade limit (default: 20)
- Maximum drawdown (-10%)
- Learning rate adjustments

### Mystic Integration
- Tesla 369 calculations
- Faerie Star patterns
- Lagos alignment timing
- Combined signal strength

## 🛠️ Maintenance

### Backup & Restore
```bash
# Manual backup
python -c "from backup_utils import backup_files; backup_files()"

# Restore latest
python -c "from backup_utils import restore_latest; restore_latest()"
```

### Model Management
```bash
# Rollback bad adjustments
python -c "from model_versioning import rollback_model; rollback_model()"

# Save version
python -c "from model_versioning import save_model_version; save_model_version('v1.2')"
```

### Performance Analysis
```bash
# Generate charts
python -c "from chart_generator import plot_performance_over_time; plot_performance_over_time()"

# Export data
python -c "from export_history import export_performance_report; export_performance_report()"
```

## 🔒 Security

### Safety Measures
- No real trades in training mode
- Automatic shutdown on excessive losses
- Rate limiting on API calls
- Input validation on all endpoints

### Monitoring
- Real-time performance tracking
- Automatic alerts on issues
- Daily health checks
- Performance plateau detection

## 📝 Usage Examples

### Start Training Mode
```python
from ai_mode_controller import AITradingController
controller = AITradingController()
controller.set_mode("training")
```

### Check AI Performance
```python
from ai_self_rating import get_ai_rating
rating = get_ai_rating()
print(f"AI Score: {rating['ai_score']}/100 - {rating['rating']}")
```

### Send Custom Alert
```python
from notifier import send_alert
send_alert("🚀 Custom trading alert message")
```

## 🐳 Docker Deployment

### Build & Run
```bash
# Build image
docker build -t mystic-ai-trader .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f mystic-ai-trader
```

### Environment Variables
```bash
# Set in docker-compose.yml or .env file
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

## 🔄 Continuous Learning

The AI system continuously learns and adapts:

1. **Performance Analysis**: Evaluates trade outcomes
2. **Strategy Adjustment**: Modifies confidence thresholds
3. **Pattern Recognition**: Learns from market patterns
4. **Mystic Integration**: Incorporates cosmic signals
5. **Self-Assessment**: Rates its own performance

## 🚨 Troubleshooting

### Common Issues
- **Import errors**: Ensure all dependencies installed
- **Database errors**: Check file permissions
- **Docker issues**: Verify Docker installation
- **API errors**: Check port availability

### Logs
- Check application logs for detailed error messages
- Use `docker-compose logs` for container issues
- Monitor AI performance through dashboard

## 📞 Support

For issues or questions:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure proper environment setup
4. Review configuration parameters

---

**Built for Windows 11 Home with Docker + Cursor IDE**
