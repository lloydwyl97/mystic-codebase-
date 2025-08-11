import numpy as np
import pandas as pd
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Tuple
import sqlite3
from scipy.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced configuration
RISK_DB = "./data/risk_optimization.db"
OPTIMIZATION_INTERVAL = 3600  # 1 hour
MAX_POSITION_SIZE = 0.2  # 20% max per position
TARGET_VOLATILITY = 0.15  # 15% target volatility
RISK_FREE_RATE = 0.02  # 2% risk-free rate


class RiskDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize risk database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                weight REAL NOT NULL,
                position_size REAL NOT NULL,
                risk_score REAL NOT NULL,
                expected_return REAL NOT NULL,
                volatility REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                portfolio_value REAL NOT NULL,
                total_risk REAL NOT NULL,
                var_95 REAL NOT NULL,
                cvar_95 REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                correlation_matrix TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stress_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                portfolio_loss REAL NOT NULL,
                worst_case_loss REAL NOT NULL,
                recovery_time_days INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def save_portfolio_weights(self, weights_data: List[Dict]):
        """Save portfolio weights to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for data in weights_data:
            cursor.execute(
                """
                INSERT INTO portfolio_weights
                (timestamp, symbol, weight, position_size, risk_score, expected_return, volatility, sharpe_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["timestamp"],
                    data["symbol"],
                    data["weight"],
                    data["position_size"],
                    data["risk_score"],
                    data["expected_return"],
                    data["volatility"],
                    data["sharpe_ratio"],
                ),
            )

        conn.commit()
        conn.close()

    def save_risk_metrics(self, metrics: Dict):
        """Save risk metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO risk_metrics
            (timestamp, portfolio_value, total_risk, var_95, cvar_95, max_drawdown, sharpe_ratio, correlation_matrix)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics["timestamp"],
                metrics["portfolio_value"],
                metrics["total_risk"],
                metrics["var_95"],
                metrics["cvar_95"],
                metrics["max_drawdown"],
                metrics["sharpe_ratio"],
                json.dumps(metrics["correlation_matrix"]),
            ),
        )

        conn.commit()
        conn.close()

    def save_stress_test(self, stress_data: Dict):
        """Save stress test results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO stress_tests
            (timestamp, scenario_name, portfolio_loss, worst_case_loss, recovery_time_days)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                stress_data["timestamp"],
                stress_data["scenario_name"],
                stress_data["portfolio_loss"],
                stress_data["worst_case_loss"],
                stress_data["recovery_time_days"],
            ),
        )

        conn.commit()
        conn.close()


def get_historical_returns(symbols: List[str], days: int = 90) -> pd.DataFrame:
    """Get historical returns for portfolio optimization"""
    returns_data = {}

    for symbol in symbols:
        try:
            # Get klines data
            url = "https://api.binance.us/api/v3/klines"
            params = {"symbol": symbol, "interval": "1d", "limit": days}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()

                prices = [float(candle[4]) for candle in data]  # Close prices
                returns = np.diff(prices) / prices[:-1]  # Calculate returns
                returns_data[symbol] = returns

        except Exception as e:
            print(f"Historical data fetch error for {symbol}: {e}")

    return pd.DataFrame(returns_data)


def calculate_portfolio_metrics(weights: np.array, returns: pd.DataFrame) -> Dict:
    """Calculate portfolio risk and return metrics"""
    # Expected returns
    expected_returns = returns.mean()
    portfolio_return = np.sum(weights * expected_returns)

    # Covariance matrix
    cov_matrix = returns.cov()
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Sharpe ratio
    sharpe_ratio = (
        (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
        if portfolio_volatility > 0
        else 0
    )

    # Value at Risk (95% confidence)
    portfolio_returns = np.dot(returns, weights)
    var_95 = np.percentile(portfolio_returns, 5)

    # Conditional Value at Risk (Expected Shortfall)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    # Maximum drawdown simulation
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "expected_return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "max_drawdown": max_drawdown,
    }


def optimize_portfolio(returns: pd.DataFrame) -> Tuple[np.array, Dict]:
    """Optimize portfolio weights using Markowitz optimization"""
    n_assets = len(returns.columns)

    # Initial weights (equal weight)
    initial_weights = np.array([1 / n_assets] * n_assets)

    # Expected returns and covariance matrix
    expected_returns = returns.mean()
    cov_matrix = returns.cov()

    # Objective function: maximize Sharpe ratio
    def objective(weights):
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (
            (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
            if portfolio_volatility > 0
            else 0
        )
        return -sharpe  # Minimize negative Sharpe ratio

    # Constraints
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
        {
            "type": "ineq",
            "fun": (lambda x: TARGET_VOLATILITY - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))),
        },  # Volatility constraint
    ]

    # Bounds: no short selling, max position size
    bounds = [(0, MAX_POSITION_SIZE) for _ in range(n_assets)]

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
        metrics = calculate_portfolio_metrics(optimal_weights, returns)
        return optimal_weights, metrics
    else:
        print(f"Optimization failed: {result.message}")
        return initial_weights, calculate_portfolio_metrics(initial_weights, returns)


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for risk analysis"""
    return returns.corr()


def run_stress_tests(weights: np.array, returns: pd.DataFrame) -> List[Dict]:
    """Run stress tests on portfolio"""
    stress_scenarios = [
        {
            "name": "market_crash",
            "shock": -0.3,
            "recovery_days": 180,
        },  # 30% market crash
        {
            "name": "volatility_spike",
            "shock": 0.5,
            "recovery_days": 90,
        },  # 50% volatility increase
        {
            "name": "correlation_breakdown",
            "shock": -0.2,  # Correlation breakdown
            "recovery_days": 120,
        },
    ]

    stress_results = []

    for scenario in stress_scenarios:
        # Simulate scenario impact
        if scenario["name"] == "market_crash":
            # Apply negative shock to all assets
            shocked_returns = returns * (1 + scenario["shock"])
        elif scenario["name"] == "volatility_spike":
            # Increase volatility
            shocked_returns = returns * (1 + np.random.normal(0, scenario["shock"], returns.shape))
        else:
            # Correlation breakdown - increase correlation
            shocked_returns = returns

        # Calculate portfolio loss
        portfolio_returns = np.dot(shocked_returns, weights)
        portfolio_loss = np.mean(portfolio_returns)
        worst_case_loss = np.percentile(portfolio_returns, 1)  # 1% worst case

        stress_results.append(
            {
                "timestamp": datetime.timezone.utcnow().isoformat(),
                "scenario_name": scenario["name"],
                "portfolio_loss": portfolio_loss,
                "worst_case_loss": worst_case_loss,
                "recovery_time_days": scenario["recovery_days"],
            }
        )

    return stress_results


def optimize_risk_enhanced():
    """Enhanced risk optimization with all features"""
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        db = RiskDatabase(RISK_DB)

        # Get historical returns
        returns_df = get_historical_returns(symbols, days=90)

        if returns_df.empty or len(returns_df) < 30:
            print("[Risk] Insufficient data for optimization")
            return

        # Optimize portfolio
        optimal_weights, portfolio_metrics = optimize_portfolio(returns_df)

        # Calculate correlation matrix
        correlation_matrix = calculate_correlation_matrix(returns_df)

        # Prepare portfolio weights data
        weights_data = []
        for i, symbol in enumerate(symbols):
            if i < len(optimal_weights):
                weights_data.append(
                    {
                        "timestamp": datetime.timezone.utcnow().isoformat(),
                        "symbol": symbol,
                        "weight": optimal_weights[i],
                        "position_size": (optimal_weights[i] * 100000),  # Assuming $100k portfolio
                        "risk_score": (1 - optimal_weights[i]),  # Inverse of weight as risk score
                        "expected_return": returns_df[symbol].mean(),
                        "volatility": returns_df[symbol].std(),
                        "sharpe_ratio": (
                            (returns_df[symbol].mean() - RISK_FREE_RATE) / returns_df[symbol].std()
                        ),
                    }
                )

        # Save portfolio weights
        db.save_portfolio_weights(weights_data)

        # Save risk metrics
        risk_metrics = {
            "timestamp": datetime.timezone.utcnow().isoformat(),
            "portfolio_value": 100000,  # Assumed portfolio value
            "total_risk": portfolio_metrics["volatility"],
            "var_95": portfolio_metrics["var_95"],
            "cvar_95": portfolio_metrics["cvar_95"],
            "max_drawdown": portfolio_metrics["max_drawdown"],
            "sharpe_ratio": portfolio_metrics["sharpe_ratio"],
            "correlation_matrix": correlation_matrix.to_dict(),
        }
        db.save_risk_metrics(risk_metrics)

        # Run stress tests
        stress_results = run_stress_tests(optimal_weights, returns_df)
        for stress_result in stress_results:
            db.save_stress_test(stress_result)

        # Print results
        print("[Risk] Portfolio Optimization Complete:")
        print(f"[Risk] Expected Return: {portfolio_metrics['expected_return']:.4f}")
        print(f"[Risk] Volatility: {portfolio_metrics['volatility']:.4f}")
        print(f"[Risk] Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.4f}")
        print(f"[Risk] VaR (95%): {portfolio_metrics['var_95']:.4f}")
        print(f"[Risk] Max Drawdown: {portfolio_metrics['max_drawdown']:.4f}")

        print("[Risk] Optimal Weights:")
        for i, symbol in enumerate(symbols):
            if i < len(optimal_weights):
                print(f"[Risk] {symbol}: {optimal_weights[i]:.3f} ({optimal_weights[i]*100:.1f}%)")

        print("[Risk] Stress Test Results:")
        for stress in stress_results:
            print(
                f"[Risk] {stress['scenario_name']}: Loss {stress['portfolio_loss']:.4f}, Worst {stress['worst_case_loss']:.4f}"
            )

    except Exception as e:
        print(f"[Risk] Enhanced optimization error: {e}")


# Main execution loop
while True:
    optimize_risk_enhanced()
    time.sleep(OPTIMIZATION_INTERVAL)
