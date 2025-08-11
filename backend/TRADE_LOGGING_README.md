# Trade Logging & Strategy Memory Engine

A complete SQLite-based trade logging and strategy evolution system for the Mystic AI Trading Platform.

## 🎯 Overview

This system provides:

- **Automatic trade logging** with SQLite database
- **Strategy performance tracking** and evaluation
- **AI-powered strategy evolution** through mutation and crossover
- **Real-time dashboard** with interactive charts
- **Discord alerts** for important events
- **Integration hooks** for existing trading systems

## 📁 File Structure

```Text
backend/
├── models.py                    # SQLite database models
├── db_logger.py                 # Database logging functions
├── reward_engine.py             # Strategy evaluation engine
├── mutator.py                   # Strategy evolution engine
├── alerts.py                    # Discord notification system
├── dashboard.py                 # Real-time web dashboard
├── trade_memory_integration.py  # Integration hooks
├── example_usage.py             # Usage examples
└── TRADE_LOGGING_README.md      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

All required packages are already in `requirements.txt`:

- SQLAlchemy (database ORM)
- Plotly (dashboard charts)
- FastAPI (web dashboard)
- Requests (Discord alerts)

### 2. Initialize the System

```python
from db_logger import init_db
from trade_memory_integration import trade_memory

# Initialize database
init_db()

# The system automatically initializes default strategies
```

### 3. Log Your First Trade

```python
from trade_memory_integration import log_trade_entry, log_trade_exit

# Log trade entry
trade_id = log_trade_entry(
    coin="BTCUSDT",
    strategy_name="Breakout_EMA",
    entry_price=45000.0,
    quantity=0.1,
    entry_reason="EMA crossover signal"
)

# Later, log trade exit
success = log_trade_exit(
    trade_id=trade_id,
    exit_price=46000.0,
    exit_reason="Take profit target reached"
)
```

### 4. Start the Dashboard

```bash
python dashboard.py
```

Visit `http://localhost:8080` to see the real-time dashboard.

## 📊 Core Components

### 1. Database Models (`models.py`)

**Strategy Table:**

- `id`: Primary key
- `name`: Strategy name (unique)
- `description`: Strategy description
- `win_rate`: Current win rate
- `avg_profit`: Average profit per trade
- `trades_executed`: Total trades executed
- `total_profit`: Cumulative profit
- `is_active`: Whether strategy is active

**Trade Table:**

- `id`: Primary key
- `coin`: Trading pair (e.g., 'BTCUSDT')
- `strategy_id`: Reference to strategy
- `entry_price`: Entry price
- `exit_price`: Exit price (null for open trades)
- `profit`: Calculated profit
- `profit_percentage`: Profit percentage
- `duration_minutes`: Trade duration
- `success`: Boolean success flag
- `entry_reason`: Reason for entry
- `exit_reason`: Reason for exit

**Strategy Performance Table:**

- Historical performance records for analysis

### 2. Database Logger (`db_logger.py`)

**Key Functions:**

- `log_trade()`: Log a complete trade
- `register_strategy()`: Register new strategy
- `get_strategy_stats()`: Get performance statistics
- `get_recent_trades()`: Get recent trade history
- `get_active_strategies()`: Get all active strategies

### 3. Reward Engine (`reward_engine.py`)

**Key Functions:**

- `evaluate_strategies()`: Evaluate all strategies
- `get_top_performers()`: Get best strategies
- `get_poor_performers()`: Get worst strategies
- `deactivate_strategy()`: Deactivate poor strategies
- `calculate_sharpe_ratio()`: Calculate risk-adjusted returns

### 4. Mutation Engine (`mutator.py`)

**Key Functions:**

- `mutate_top_strategies()`: Create mutations of best strategies
- `crossover_strategies()`: Combine two strategies
- `create_random_strategy()`: Create random exploration strategies
- `evolve_strategy_population()`: Run full evolution cycle
- `cleanup_poor_strategies()`: Remove poor performers

### 5. Alerts System (`alerts.py`)

**Key Functions:**

- `alert_trade_execution()`: Alert on trade completion
- `alert_strategy_mutation()`: Alert on new mutations
- `alert_evolution_cycle()`: Alert on evolution completion
- `alert_daily_summary()`: Daily performance summary
- `alert_system_health()`: System health alerts

### 6. Dashboard (`dashboard.py`)

**Features:**

- Real-time profit charts
- Strategy performance comparison
- Recent trades table
- Interactive API endpoints
- Auto-refresh every 30 seconds

**API Endpoints:**

- `GET /api/stats` - Dashboard statistics
- `GET /api/profit-chart` - Profit chart data
- `GET /api/strategy-chart` - Strategy performance
- `GET /api/trades-table` - Recent trades
- `GET /api/strategies` - All strategies
- `GET /api/top-performers` - Top performers

### 7. Integration (`trade_memory_integration.py`)

**Key Functions:**

- `log_trade_entry()`: Log trade entry
- `log_trade_exit()`: Log trade exit
- `get_strategy_performance()`: Get strategy stats
- `force_evaluation()`: Force strategy evaluation
- `force_evolution()`: Force evolution cycle

## 🔧 Integration with Existing Systems

### Hook into Your Trading Engine

```python
# In your existing trading bot
from trade_memory_integration import log_trade_entry, log_trade_exit

class YourTradingBot:
    def execute_buy(self, coin, price, strategy_name):
        # Your existing buy logic
        order = self.broker.buy(coin, price)
        
        # Log the trade
        trade_id = log_trade_entry(
            coin=coin,
            strategy_name=strategy_name,
            entry_price=price,
            entry_reason="Buy signal triggered"
        )
        
        return order, trade_id
    
    def execute_sell(self, trade_id, price):
        # Your existing sell logic
        order = self.broker.sell(coin, price)
        
        # Log the exit
        log_trade_exit(
            trade_id=trade_id,
            exit_price=price,
            exit_reason="Sell signal triggered"
        )
        
        return order
```

### Automatic Evolution

The system automatically:

- Evaluates strategies every 100 trades
- Runs evolution cycles every 500 trades
- Sends Discord alerts for important events
- Cleans up poor performing strategies

## 📈 Strategy Evolution

### How Evolution Works

1. **Evaluation**: System evaluates all strategies based on:
   - Win rate
   - Average profit
   - Total profit
   - Sharpe ratio
   - Maximum drawdown

2. **Selection**: Top performers are selected for evolution

3. **Mutation**: Creates variations of best strategies:
   - Parameter tweaks
   - Indicator combinations
   - Risk adjustments

4. **Crossover**: Combines two good strategies

5. **Random**: Creates random strategies for exploration

6. **Cleanup**: Removes poor performers

### Evolution Triggers

- **Every 100 trades**: Strategy evaluation
- **Every 500 trades**: Full evolution cycle
- **Manual**: Call `force_evolution()`

## 🔔 Discord Alerts

### Setup Discord Webhook

1. Create a Discord webhook in your server
2. Add to `.env` file:

```Text
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url
```

### Alert Types

- **Trade Execution**: Every completed trade
- **Strategy Mutation**: New strategy created
- **Evolution Cycle**: Evolution results
- **Daily Summary**: End-of-day performance
- **System Health**: System status alerts

## 📊 Dashboard Features

### Real-time Charts

1. **Profit Chart**: Cumulative profit over time
2. **Strategy Performance**: Win rate vs profit comparison
3. **Recent Trades**: Live trade table
4. **Performance Stats**: Key metrics

### Auto-refresh

- Dashboard refreshes every 30 seconds
- Charts update automatically
- No page reload required

## 🧪 Testing the System

### Run the Example

```bash
python example_usage.py
```

This will:

1. Simulate 10 trades
2. Show performance statistics
3. Run evolution cycle
4. Display dashboard information

### Manual Testing

```python
from trade_memory_integration import *

# Test trade logging
trade_id = log_trade_entry("BTCUSDT", "Test_Strategy", 45000.0)
log_trade_exit(trade_id, 46000.0, "Test exit")

# Test evaluation
force_evaluation()

# Test evolution
force_evolution()

# Check performance
stats = get_strategy_performance("Test_Strategy")
print(stats)
```

## 🔧 Configuration

### Environment Variables

```bash
# Discord webhook (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Database path (default: trading_memory.db)
TRADING_DB_PATH=./trading_memory.db

# Evaluation intervals
EVALUATION_INTERVAL=100  # Evaluate every N trades
EVOLUTION_INTERVAL=500   # Evolve every N trades
```

### Customization

**Strategy Evaluation Criteria:**

```python
# In reward_engine.py
def evaluate_strategies(min_trades=5, days=7):
    # Customize evaluation logic
```

**Evolution Parameters:**

```python
# In mutator.py
def evolve_strategy_population(mutation_rate=0.3, crossover_rate=0.2, random_rate=0.1):
    # Customize evolution rates
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check file permissions
   - Ensure SQLite is available

2. **Discord Alerts Not Working**
   - Verify webhook URL
   - Check network connectivity

3. **Dashboard Not Loading**
   - Check port 8080 is available
   - Verify all dependencies installed

4. **Import Errors**
   - Ensure all files are in the same directory
   - Check Python path

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed logs
```

## 📈 Performance Monitoring

### Key Metrics

- **Win Rate**: Percentage of profitable trades
- **Average Profit**: Mean profit per trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Total Profit**: Cumulative profit

### Monitoring Dashboard

The dashboard provides real-time monitoring of:

- Overall system performance
- Individual strategy performance
- Trade history and patterns
- Evolution progress

## 🔮 Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - ML-based strategy generation
   - Predictive performance modeling

2. **Advanced Analytics**
   - Risk metrics (VaR, CVaR)
   - Correlation analysis
   - Market regime detection

3. **Multi-Exchange Support**
   - Cross-exchange arbitrage
   - Portfolio optimization

4. **Backtesting Engine**
   - Historical strategy testing
   - Walk-forward analysis

5. **API Extensions**
   - REST API for external access
   - WebSocket real-time data

## 📞 Support

For issues or questions:

1. Check the troubleshooting section
2. Review the example code
3. Check the logs for error messages
4. Test with the example script

## 📄 License

This system is part of the Mystic AI Trading Platform.
Licensed under the MIT Licence
