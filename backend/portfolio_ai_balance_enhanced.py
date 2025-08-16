import numpy as np
import pandas as pd
import time
import json
import sqlite3
import requests
from datetime import datetime
from typing import Dict, List, Tuple
from scipy.optimize import minimize

# Enhanced configuration
BALANCE_DB = "./data/portfolio_balance.db"
BALANCE_INTERVAL = 1800  # 30 minutes
REBALANCE_THRESHOLD = 0.05  # 5% deviation triggers rebalancing
MAX_ASSETS = 10  # Maximum number of assets in portfolio
RISK_FREE_RATE = 0.02  # 2% risk-free rate


class BalanceDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize balance database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                asset_allocations TEXT NOT NULL,
                risk_metrics TEXT NOT NULL,
                rebalance_needed BOOLEAN NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rebalance_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                reason TEXT NOT NULL,
                performance_impact REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS asset_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                weight REAL NOT NULL,
                return_1d REAL NOT NULL,
                return_7d REAL NOT NULL,
                return_30d REAL NOT NULL,
                volatility REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                correlation_btc REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_portfolio_snapshot(self, snapshot: Dict):
        """Save portfolio snapshot to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO portfolio_snapshots
            (timestamp, total_value, cash_balance, asset_allocations, risk_metrics, rebalance_needed)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                snapshot["timestamp"],
                snapshot["total_value"],
                snapshot["cash_balance"],
                json.dumps(snapshot["asset_allocations"]),
                json.dumps(snapshot["risk_metrics"]),
                snapshot["rebalance_needed"],
            ),
        )

        conn.commit()
        conn.close()

    def save_rebalance_action(self, action: Dict):
        """Save rebalance action to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO rebalance_actions
            (timestamp, action_type, symbol, quantity, price, reason, performance_impact)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                action["timestamp"],
                action["action_type"],
                action["symbol"],
                action["quantity"],
                action["price"],
                action["reason"],
                action.get("performance_impact", 0),
            ),
        )

        conn.commit()
        conn.close()

    def save_asset_performance(self, performance: Dict):
        """Save asset performance to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO asset_performance
            (timestamp, symbol, weight, return_1d, return_7d, return_30d, volatility, sharpe_ratio, correlation_btc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                performance["timestamp"],
                performance["symbol"],
                performance["weight"],
                performance["return_1d"],
                performance["return_7d"],
                performance["return_30d"],
                performance["volatility"],
                performance["sharpe_ratio"],
                performance["correlation_btc"],
            ),
        )

        conn.commit()
        conn.close()


def get_market_data(symbols: List[str]) -> Dict[str, Dict]:
    """Get comprehensive market data for portfolio analysis"""
    market_data = {}

    try:
        # Get 24h ticker data
        response = requests.get("https://api.binance.us/api/v3/ticker/24hr", timeout=10)
        if response.status_code == 200:
            data = response.json()

            for item in data:
                if item["symbol"] in symbols:
                    market_data[item["symbol"]] = {
                        "price": float(item["lastPrice"]),
                        "change_24h": float(item["priceChangePercent"]),
                        "volume": float(item["volume"]),
                        "high_24h": float(item["highPrice"]),
                        "low_24h": float(item["lowPrice"]),
                    }

    except Exception as e:
        print(f"Market data fetch error: {e}")

    return market_data


def get_historical_returns(symbol: str, days: int = 30) -> pd.Series:
    """Get historical returns for asset analysis"""
    try:
        url = "https://api.binance.us/api/v3/klines"
        params = {"symbol": symbol, "interval": "1d", "limit": days}

        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()

            prices = [float(candle[4]) for candle in data]  # Close prices
            returns = pd.Series(prices).pct_change().dropna()

            return returns

    except Exception as e:
        print(f"Historical data fetch error for {symbol}: {e}")

    return pd.Series()


def calculate_asset_metrics(symbol: str, current_weight: float) -> Dict:
    """Calculate comprehensive asset metrics"""
    returns = get_historical_returns(symbol, days=30)

    if returns.empty:
        return {
            "symbol": symbol,
            "weight": current_weight,
            "return_1d": 0,
            "return_7d": 0,
            "return_30d": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "correlation_btc": 0,
        }

    # Calculate returns for different periods
    return_1d = returns.iloc[-1] if len(returns) > 0 else 0
    return_7d = returns.tail(7).mean() if len(returns) >= 7 else 0
    return_30d = returns.mean()

    # Calculate volatility
    volatility = returns.std()

    # Calculate Sharpe ratio
    sharpe_ratio = (return_30d - RISK_FREE_RATE / 365) / volatility if volatility > 0 else 0

    # Calculate correlation with BTC
    btc_returns = get_historical_returns("BTCUSDT", days=30)
    if not btc_returns.empty and len(returns) == len(btc_returns):
        correlation_btc = returns.corr(btc_returns)
    else:
        correlation_btc = 0

    return {
        "symbol": symbol,
        "weight": current_weight,
        "return_1d": return_1d,
        "return_7d": return_7d,
        "return_30d": return_30d,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "correlation_btc": correlation_btc,
    }


def optimize_portfolio_weights(
    current_weights: Dict[str, float], asset_metrics: List[Dict]
) -> Dict[str, float]:
    """Optimize portfolio weights using modern portfolio theory"""
    try:
        symbols = list(current_weights.keys())
        n_assets = len(symbols)

        if n_assets == 0:
            return current_weights

        # Get historical returns for all assets
        returns_data = {}
        for symbol in symbols:
            returns = get_historical_returns(symbol, days=30)
            if not returns.empty:
                returns_data[symbol] = returns

        if len(returns_data) < 2:
            return current_weights

        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        if len(returns_df) < 10:
            return current_weights

        # Calculate expected returns and covariance matrix
        expected_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        # Initial weights
        initial_weights = np.array([current_weights.get(symbol, 0) for symbol in symbols])

        # Objective function: maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (
                (portfolio_return - RISK_FREE_RATE / 365) / portfolio_volatility
                if portfolio_volatility > 0
                else 0
            )
            return -sharpe  # Minimize negative Sharpe ratio

        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Weights sum to 1

        # Bounds: no short selling, max 30% per asset
        bounds = [(0, 0.3) for _ in range(n_assets)]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            optimal_weights = result.x
            return {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
        else:
            print(f"Portfolio optimization failed: {result.message}")
            return current_weights

    except Exception as e:
        print(f"Portfolio optimization error: {e}")
        return current_weights


def calculate_portfolio_risk_metrics(weights: Dict[str, float], asset_metrics: List[Dict]) -> Dict:
    """Calculate portfolio risk metrics"""
    try:
        # Calculate weighted metrics
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {
                "total_volatility": 0,
                "portfolio_sharpe": 0,
                "max_drawdown_estimate": 0,
                "var_95": 0,
                "diversification_score": 0,
            }

        # Weighted returns and volatility
        weighted_return = sum(
            weights[symbol] * metrics["return_30d"]
            for symbol, metrics in zip(weights.keys(), asset_metrics)
            if symbol in weights
        )
        weighted_volatility = sum(
            weights[symbol] * metrics["volatility"]
            for symbol, metrics in zip(weights.keys(), asset_metrics)
            if symbol in weights
        )

        # Portfolio Sharpe ratio
        portfolio_sharpe = (
            (weighted_return - RISK_FREE_RATE / 365) / weighted_volatility
            if weighted_volatility > 0
            else 0
        )

        # Estimate max drawdown (simplified)
        max_drawdown_estimate = weighted_volatility * 2.5  # Rough estimate

        # Value at Risk (95% confidence)
        var_95 = weighted_volatility * 1.645  # Normal distribution assumption

        # Diversification score (inverse of concentration)
        concentration = sum(w**2 for w in weights.values())
        diversification_score = 1 - concentration

        return {
            "total_volatility": weighted_volatility,
            "portfolio_sharpe": portfolio_sharpe,
            "max_drawdown_estimate": max_drawdown_estimate,
            "var_95": var_95,
            "diversification_score": diversification_score,
        }

    except Exception as e:
        print(f"Risk metrics calculation error: {e}")
        return {
            "total_volatility": 0,
            "portfolio_sharpe": 0,
            "max_drawdown_estimate": 0,
            "var_95": 0,
            "diversification_score": 0,
        }


def check_rebalance_needed(
    current_weights: Dict[str, float], target_weights: Dict[str, float]
) -> Tuple[bool, List[Dict]]:
    """Check if rebalancing is needed and generate actions"""
    rebalance_actions = []
    rebalance_needed = False

    for symbol in set(current_weights.keys()) | set(target_weights.keys()):
        current_weight = current_weights.get(symbol, 0)
        target_weight = target_weights.get(symbol, 0)

        deviation = abs(current_weight - target_weight)

        if deviation > REBALANCE_THRESHOLD:
            rebalance_needed = True

            if target_weight > current_weight:
                action_type = "BUY"
                quantity = target_weight - current_weight
                reason = f"Rebalance: Increase allocation to {target_weight:.3f}"
            else:
                action_type = "SELL"
                quantity = current_weight - target_weight
                reason = f"Rebalance: Decrease allocation from {current_weight:.3f}"

            rebalance_actions.append(
                {
                    "timestamp": datetime.timezone.utcnow().isoformat(),
                    "action_type": action_type,
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": 0,  # Would be filled with actual price
                    "reason": reason,
                    "performance_impact": 0,
                }
            )

    return rebalance_needed, rebalance_actions


def balance_portfolio_enhanced():
    """Enhanced portfolio balancing with all features"""
    try:
        db = BalanceDatabase(BALANCE_DB)

        # Current portfolio state (simulated)
        current_portfolio = {
            "BTCUSDT": 0.4,
            "ETHUSDT": 0.3,
            "ADAUSDT": 0.15,
            "DOTUSDT": 0.1,
            "LINKUSDT": 0.05,
        }

        total_value = 100000  # $100k portfolio
        cash_balance = 5000  # $5k cash

        # Get market data
        symbols = list(current_portfolio.keys())
        get_market_data(symbols)

        # Calculate asset metrics
        asset_metrics = []
        for symbol, weight in current_portfolio.items():
            metrics = calculate_asset_metrics(symbol, weight)
            asset_metrics.append(metrics)
            db.save_asset_performance(metrics)

        # Optimize portfolio weights
        target_weights = optimize_portfolio_weights(current_portfolio, asset_metrics)

        # Calculate risk metrics
        risk_metrics = calculate_portfolio_risk_metrics(target_weights, asset_metrics)

        # Check if rebalancing is needed
        rebalance_needed, rebalance_actions = check_rebalance_needed(
            current_portfolio, target_weights
        )

        # Save portfolio snapshot
        snapshot = {
            "timestamp": datetime.timezone.utcnow().isoformat(),
            "total_value": total_value,
            "cash_balance": cash_balance,
            "asset_allocations": {
                "current": current_portfolio,
                "target": target_weights,
            },
            "risk_metrics": risk_metrics,
            "rebalance_needed": rebalance_needed,
        }
        db.save_portfolio_snapshot(snapshot)

        # Save rebalance actions
        for action in rebalance_actions:
            db.save_rebalance_action(action)

        # Print results
        print("[Balance] Portfolio Analysis Complete:")
        print(f"[Balance] Total Value: ${total_value:,.2f}")
        print(f"[Balance] Cash Balance: ${cash_balance:,.2f}")
        print(f"[Balance] Portfolio Sharpe: {risk_metrics['portfolio_sharpe']:.3f}")
        print(f"[Balance] Volatility: {risk_metrics['total_volatility']:.3f}")
        print(f"[Balance] Diversification: {risk_metrics['diversification_score']:.3f}")

        print("[Balance] Current Allocations:")
        for symbol, weight in current_portfolio.items():
            print(f"[Balance] {symbol}: {weight:.3f} ({weight*100:.1f}%)")

        print("[Balance] Target Allocations:")
        for symbol, weight in target_weights.items():
            print(f"[Balance] {symbol}: {weight:.3f} ({weight*100:.1f}%)")

        if rebalance_needed:
            print(f"[Balance] Rebalancing Required: {len(rebalance_actions)} actions")
            for action in rebalance_actions:
                print(
                    f"[Balance] {action['action_type']} {action['symbol']}: {action['quantity']:.3f} - {action['reason']}"
                )
        else:
            print("[Balance] Portfolio is balanced")

    except Exception as e:
        print(f"[Balance] Enhanced balancing error: {e}")


# Main execution loop
while True:
    balance_portfolio_enhanced()
    time.sleep(BALANCE_INTERVAL)


