# Mystic Trading Platform

A comprehensive autonomous crypto trading platform with real-time signals, AI strategies, and live exchange execution.

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Redis Server
- Node.js (for frontend)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Mystic-Codebase
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install development dependencies**

   ```bash
   pip install -r backend/requirements-dev.txt
   ```

4. **Start Redis Server**

   ```bash
   # Windows
   start-redis.bat

   # Or manually start Redis server
   ```

5. **Start the backend**

   ```bash
   # Windows
   scripts/start-backend.bat

   # Or manually
   cd backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Start the frontend**

   ```bash
   # Windows
   scripts/start-frontend.bat

   # Or manually
   cd frontend
   npm install
   npm start
   ```

## 📁 Project Structure

```Text
Mystic-Codebase/
├── backend/                 # FastAPI backend application
│   ├── ai/                 # AI and machine learning modules
│   ├── endpoints/          # API endpoints
│   ├── services/           # Business logic services
│   ├── middleware/         # Custom middleware
│   ├── utils/              # Utility functions
│   └── tests/              # Backend tests
├── frontend/               # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── utils/          # Frontend utilities
│   └── public/             # Static assets
├── docs/                   # Documentation
├── scripts/                # Setup and deployment scripts
├── logs/                   # Application logs
└── requirements.txt        # Python dependencies
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=sqlite:///./mystic_trading.db

# Redis
REDIS_URL=redis://localhost:6379

# API Keys (add your own)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET_KEY=your_coinbase_secret_key

# AI Services
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Security
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret
```

## 🧠 AI Features

- **Strategy Generation**: AI-powered trading strategy creation
- **Market Analysis**: Real-time market data analysis
- **Risk Management**: Automated risk assessment and management
- **Portfolio Optimization**: AI-driven portfolio rebalancing
- **Signal Generation**: Advanced trading signal algorithms

## 📊 Trading Features

- **Multi-Exchange Support**: Binance, Coinbase, and more
- **Real-time Data**: Live market data and price feeds
- **Automated Trading**: Fully automated trading execution
- **Risk Management**: Built-in risk controls and limits
- **Performance Analytics**: Comprehensive trading analytics

## 🛠️ Development

### Code Quality

```bash
# Run quality checks
python backend/run_quality_checks.py

# Format code
black backend/
isort backend/

# Type checking
mypy backend/

# Security scanning
bandit -r backend/
```

### Testing

```bash
# Run tests
pytest backend/tests/

# Run with coverage
pytest --cov=backend backend/tests/
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Manual Deployment

```bash
# Start all services
scripts/start-all.bat
```

## 📚 Documentation

- [API Documentation](docs/API_KEYS_SETUP_GUIDE.md)
- [AI Strategy Guide](docs/AI_STRATEGY_README.md)
- [Deployment Guide](docs/LIVE_DEPLOYMENT_README.md)
- [Trading Bots Guide](docs/TRADING_BOTS_README.md)
- [Architecture Overview](docs/MODULAR_STRUCTURE_README.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run quality checks
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk.

## 🆘 Support

For support and questions:

- Check the documentation in the `docs/` folder
- Review the logs in the `logs/` directory
- Create an issue on GitHub

---

**Built with ❤️ using FastAPI, React, and Python 3.11**
