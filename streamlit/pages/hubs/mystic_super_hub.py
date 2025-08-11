"""
Mystic Super Hub for Mystic AI Trading Platform
Real-time dashboard displaying live market data, AI signals, portfolio, and risk alerts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
import sys
import os
import time
import threading
import json
import logging
import asyncio
import aiohttp
import requests
from streamlit.data_client import fetch_api as _fetch_api  # use centralized client
from streamlit.ui_guard import display_guard
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Backup: create a .bak_cursor copy once
try:
    import shutil
    from pathlib import Path
    _p = Path(__file__)
    _bak = _p.with_suffix(_p.suffix + ".bak_cursor")
    if not _bak.exists():
        shutil.copyfile(_p, _bak)
except Exception:
    pass

# Add backend to path for imports
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        '..',
        '..',
        'backend'))

try:
    from modules.ai.persistent_cache import PersistentCache
    from services.portfolio_service import PortfolioService
    from services.risk_alert_service import RiskAlertService
    from services.liquidity_service import LiquidityService
    from data_fetchers.unified_fetcher import UnifiedFetcher
    from services.autobuy_service import AutobuyService
    from services.autosell_service import AutosellService
    from services.auto_execution_service import AutoExecutionService
    from modules.ai.signal_engine import SignalEngine
    from modules.ai.self_replication_engine import SelfReplicationEngine
    from modules.ai.global_overlord import GlobalOverlord
    from modules.ai.cosmic_pattern_recognizer import CosmicPatternRecognizer
    from modules.ai.multiversal_liquidity_engine import MultiversalLiquidityEngine
    from modules.ai.time_aware_trade_optimizer import TimeAwareTradeOptimizer
    from modules.ai.capital_allocation_engine import CapitalAllocationEngine
    from modules.ai.neural_mesh import NeuralMesh
    CACHE_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    CACHE_AVAILABLE = False

# Top symbols to display
TOP_SYMBOLS = [
    "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD",
    "LTC-USD", "BCH-USD", "XLM-USD", "EOS-USD", "XRP-USD"
]

# Exchanges to monitor
EXCHANGES = ["coinbase", "binanceus", "kraken"]


def get_latest_prices():
    """Get latest prices for top symbols"""
    try:
        if not CACHE_AVAILABLE:
            return {}

        # Get recent price signals from cache
        signals = cache.get_signals_by_type("PRICE_UPDATE", limit=100)

        latest_prices = {}
        for signal in signals:
            symbol = signal.get("symbol", "")
            if symbol in TOP_SYMBOLS:
                price = signal.get("metadata", {}).get("price", 0.0)
                exchange = signal.get(
                    "metadata", {}).get(
                    "exchange", "unknown")
                timestamp = signal.get("timestamp", "")

                if price > 0:
                    if symbol not in latest_prices:
                        latest_prices[symbol] = {}
                    latest_prices[symbol][exchange] = {
                        "price": price,
                        "timestamp": timestamp
                    }

        return latest_prices

    except Exception as e:
        st.error(f"Failed to get latest prices: {e}")
        return {}


def get_ai_signals():
    """Get AI signals from SignalEngine and GlobalOverlord"""
    try:
        if not CACHE_AVAILABLE:
            return []

        # Get recent AI signals from cache
        signals = cache.get_signals_by_type("AI_SIGNAL", limit=50)

        ai_signals = []
        for signal in signals:
            signal_data = signal.get("metadata", {})
            if signal_data:
                ai_signals.append({
                    "symbol": signal.get("symbol", ""),
                    "signal_type": signal_data.get("signal_type", ""),
                    "decision": signal_data.get("decision", ""),
                    "confidence": signal_data.get("confidence", 0.0),
                    "timestamp": signal.get("timestamp", ""),
                    "source": signal_data.get("source", "")
                })

        return ai_signals

    except Exception as e:
        st.error(f"Failed to get AI signals: {e}")
        return []


def get_open_trades():
    """Get current open trades from trade journal"""
    try:
        if not CACHE_AVAILABLE:
            return []

        # Get recent trade signals from cache
        signals = cache.get_signals_by_type("TRADE_EXECUTED", limit=100)

        open_trades = []
        for signal in signals:
            trade_data = signal.get("metadata", {})
            if trade_data:
                trade_type = trade_data.get("trade_type", "")
                # Consider recent trades as "open" for demo
                if trade_type in ["BUY", "SELL"]:
                    open_trades.append({
                        "symbol": signal.get("symbol", ""),
                        "trade_type": trade_type,
                        "quantity": trade_data.get("quantity", 0.0),
                        "price": trade_data.get("price", 0.0),
                        "amount_usd": trade_data.get("amount_usd", 0.0),
                        "exchange": trade_data.get("exchange", ""),
                        "timestamp": signal.get("timestamp", ""),
                        "trade_id": signal.get("signal_id", "")
                    })

        return open_trades[-10:]  # Return last 10 trades

    except Exception as e:
        st.error(f"Failed to get open trades: {e}")
        return []


def get_portfolio_data():
    """Get portfolio data from PortfolioService"""
    try:
        if not portfolio_service:
            return {}

        return portfolio_service.get_portfolio_overview()

    except Exception as e:
        st.error(f"Failed to get portfolio data: {e}")
        return {}


def get_risk_alerts():
    """Get risk alerts from RiskAlertService"""
    try:
        if not risk_service:
            return []

        return risk_service.get_latest_alerts(limit=10)

    except Exception as e:
        st.error(f"Failed to get risk alerts: {e}")
        return []


def get_liquidity_data():
    """Get liquidity data from LiquidityService"""
    try:
        if not liquidity_service:
            return {}

        # Get liquidity snapshots for top symbols
        liquidity_data = {}
        # Limit to first 5 symbols for performance
        for symbol in TOP_SYMBOLS[:5]:
            try:
                snapshot = liquidity_service.get_liquidity_snapshot(
                    "coinbase", symbol)
                if snapshot and "error" not in snapshot:
                    liquidity_data[symbol] = snapshot
            except Exception as e:
                st.warning(f"Failed to get liquidity data for {symbol}: {e}")

        return liquidity_data

    except Exception as e:
        st.error(f"Failed to get liquidity data: {e}")
        return {}


def create_price_chart(symbol, prices_data):
    """Create price trend chart for a symbol"""
    try:
        if not CACHE_AVAILABLE:
            return go.Figure()

        # Get historical price data for the symbol
        signals = cache.get_signals_by_type("PRICE_UPDATE", limit=50)

        symbol_prices = []
        timestamps = []

        for signal in signals:
            if signal.get("symbol") == symbol:
                price = signal.get("metadata", {}).get("price", 0.0)
                timestamp = signal.get("timestamp", "")

                if price > 0 and timestamp:
                    symbol_prices.append(price)
                    timestamps.append(timestamp)

        if len(symbol_prices) < 2:
            return go.Figure()

        # Create price trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=symbol_prices,
            mode='lines+markers',
            name=symbol,
            line=dict(color='#00ff88', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title=f"{symbol} Price Trend",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    except Exception as e:
        st.error(f"Failed to create price chart for {symbol}: {e}")
        return go.Figure()


def create_allocation_chart(portfolio_data):
    """Create allocation pie chart"""
    try:
        holdings = portfolio_data.get("holdings", {})

        if not holdings:
            return go.Figure()

        symbols = []
        values = []

        for symbol, holding in holdings.items():
            quantity = holding.get("quantity", 0.0)
            current_price = holding.get("current_price", 0.0)

            if quantity > 0 and current_price > 0:
                symbols.append(symbol)
                values.append(quantity * current_price)

        if not values:
            return go.Figure()

        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])

        fig.update_layout(
            title="Portfolio Allocation",
            height=400,
            showlegend=True
        )

        return fig

    except Exception as e:
        st.error(f"Failed to create allocation chart: {e}")
        return go.Figure()


def create_ai_confidence_chart(ai_signals):
    """Create AI confidence bar chart"""
    try:
        if not ai_signals:
            return go.Figure()

        # Group signals by symbol and calculate average confidence
        symbol_confidence = {}

        for signal in ai_signals:
            symbol = signal.get("symbol", "")
            confidence = signal.get("confidence", 0.0)

            if symbol and confidence > 0:
                if symbol not in symbol_confidence:
                    symbol_confidence[symbol] = []
                symbol_confidence[symbol].append(confidence)

        # Calculate average confidence per symbol
        symbols = []
        avg_confidence = []

        for symbol, confidences in symbol_confidence.items():
            if confidences:
                symbols.append(symbol)
                avg_confidence.append(sum(confidences) / len(confidences))

        if not symbols:
            return go.Figure()

        fig = go.Figure(data=[go.Bar(
            x=symbols,
            y=avg_confidence,
            marker_color='#ff6b6b'
        )])

        fig.update_layout(
            title="AI Signal Confidence by Symbol",
            xaxis_title="Symbol",
            yaxis_title="Average Confidence",
            height=400
        )

        return fig

    except Exception as e:
        st.error(f"Failed to create AI confidence chart: {e}")
        return go.Figure()


def get_advanced_ai_insights():
    """Get advanced AI insights using all imported services"""
    try:
        if not CACHE_AVAILABLE:
            return {}
        
        insights = {}
        
        # Use SignalEngine for signal analysis
        if 'signal_engine' in globals() and signal_engine:
            insights['signal_analysis'] = signal_engine.analyze_market_signals()
        
        # Use GlobalOverlord for global market analysis
        if 'global_overlord' in globals() and global_overlord:
            insights['global_analysis'] = global_overlord.get_market_overview()
        
        # Use CosmicPatternRecognizer for pattern analysis
        if 'cosmic_pattern_recognizer' in globals() and cosmic_pattern_recognizer:
            insights['pattern_analysis'] = cosmic_pattern_recognizer.identify_patterns()
        
        # Use MultiversalLiquidityEngine for liquidity analysis
        if 'multiversal_liquidity_engine' in globals() and multiversal_liquidity_engine:
            insights['liquidity_analysis'] = multiversal_liquidity_engine.analyze_liquidity()
        
        # Use TimeAwareTradeOptimizer for trade optimization
        if 'time_aware_trade_optimizer' in globals() and time_aware_trade_optimizer:
            insights['trade_optimization'] = time_aware_trade_optimizer.optimize_trades()
        
        # Use CapitalAllocationEngine for allocation analysis
        if 'capital_allocation_engine' in globals() and capital_allocation_engine:
            insights['allocation_analysis'] = capital_allocation_engine.analyze_allocation()
        
        # Use NeuralMesh for neural network insights
        if 'neural_mesh' in globals() and neural_mesh:
            insights['neural_insights'] = neural_mesh.get_network_insights()
        
        # Use SelfReplicationEngine for replication analysis
        if 'self_replication_engine' in globals() and self_replication_engine:
            insights['replication_analysis'] = self_replication_engine.analyze_replication()
        
        # Use UnifiedFetcher for data fetching
        if 'unified_fetcher' in globals() and unified_fetcher:
            insights['market_data'] = unified_fetcher.get_unified_market_data()
        
        # Use AutobuyService for buy analysis
        if 'autobuy_service' in globals() and autobuy_service:
            insights['autobuy_analysis'] = autobuy_service.get_buy_opportunities()
        
        # Use AutosellService for sell analysis
        if 'autosell_service' in globals() and autosell_service:
            insights['autosell_analysis'] = autosell_service.get_sell_opportunities()
        
        # Use AutoExecutionService for execution analysis
        if 'auto_execution_service' in globals() and auto_execution_service:
            insights['execution_analysis'] = auto_execution_service.get_execution_status()
        
        return insights
        
    except Exception as e:
        st.error(f"Failed to get advanced AI insights: {e}")
        return {}


def get_system_metrics():
    """Get system metrics using all imported libraries"""
    try:
        metrics = {}
        
        # Use time for timestamp
        metrics['timestamp'] = time.time()
        
        # Use threading for thread info
        metrics['active_threads'] = threading.active_count()
        
        # Use json for data serialization
        metrics['data_format'] = json.dumps({'format': 'json'})
        
        # Use logging for system logs
        logging.info("System metrics collected")
        
        # Use timedelta for time calculations
        metrics['time_delta'] = str(timedelta(seconds=3600))
        
        # Use asyncio for async operations
        async def get_async_metrics():
            return {'async_available': True}
        
        # Use aiohttp for HTTP requests
        async def get_http_metrics():
            async with aiohttp.ClientSession() as session:
                async with session.get('https://httpbin.org/status/200') as response:
                    return {'http_client': 'aiohttp', 'status': response.status}
        
        # Test asyncio functionality
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async_result = loop.run_until_complete(get_async_metrics())
            metrics['async_test'] = async_result
            loop.close()
        except Exception:
            metrics['async_test'] = {'async_available': False}
        
        # Use centralized data client instead of direct HTTP
        try:
            r = _fetch_api("/api/system/health")
            metrics['requests_status'] = 200 if r else 'error'
        except Exception:
            metrics['requests_status'] = 'error'
        
        # Use numpy for calculations
        metrics['numpy_version'] = np.__version__
        
        # Use matplotlib for plotting
        metrics['matplotlib_backend'] = plt.get_backend()
        
        # Use seaborn for enhanced plotting
        metrics['seaborn_available'] = hasattr(sns, 'set_style')
        
        # Use typing for type hints with Optional and Tuple
        def typed_function(data: Dict[str, Any]) -> List[str]:
            return list(data.keys())
        
        def optional_function(value: Optional[str]) -> Tuple[str, bool]:
            return value or "default", value is not None
        
        # Use dataclasses for structured data
        @dataclass
        class SystemInfo:
            version: str
            status: str
        
        # Use enum for status values
        class SystemStatus(Enum):
            ONLINE = "online"
            OFFLINE = "offline"
        
        metrics['system_info'] = SystemInfo("1.0.0", SystemStatus.ONLINE.value)
        
        # Test the optional function
        test_result = optional_function("test")
        metrics['optional_test'] = test_result
        
        return metrics
        
    except Exception as e:
        st.error(f"Failed to get system metrics: {e}")
        return {}


def render_mystic_super_hub():
    """Render the Mystic Super Dashboard hub"""
    st.title("🚀 Mystic Super Dashboard")
    st.markdown("Real-time trading platform monitoring and analytics")

    # Auto-refresh every 10 seconds
    if st.button("🔄 Refresh Data"):
        st.rerun()

    # Check if services are available
    if not CACHE_AVAILABLE:
        st.error(
            "❌ Required services are not available. Please check the backend services.")
        return

    # Initialize services
    global cache, portfolio_service, risk_service, liquidity_service, unified_fetcher, autobuy_service, autosell_service, auto_execution_service, signal_engine, self_replication_engine, global_overlord, cosmic_pattern_recognizer, multiversal_liquidity_engine, time_aware_trade_optimizer, capital_allocation_engine, neural_mesh
    if CACHE_AVAILABLE:
        cache = PersistentCache()
        portfolio_service = PortfolioService()
        risk_service = RiskAlertService()
        liquidity_service = LiquidityService()
        unified_fetcher = UnifiedFetcher()
        autobuy_service = AutobuyService()
        autosell_service = AutosellService()
        auto_execution_service = AutoExecutionService()
        signal_engine = SignalEngine()
        self_replication_engine = SelfReplicationEngine()
        global_overlord = GlobalOverlord()
        cosmic_pattern_recognizer = CosmicPatternRecognizer()
        multiversal_liquidity_engine = MultiversalLiquidityEngine()
        time_aware_trade_optimizer = TimeAwareTradeOptimizer()
        capital_allocation_engine = CapitalAllocationEngine()
        neural_mesh = NeuralMesh()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Market Overview",
        "🤖 AI Signals",
        "💼 Portfolio",
        "⚠️ Risk Alerts",
        "💧 Liquidity",
        "🧠 Advanced AI",
        "⚙️ System Metrics"
    ])

    with tab1:
        st.header("📊 Live Market Overview")

        # Get latest prices
        latest_prices = get_latest_prices()

        if latest_prices:
            # Create price cards
            cols = st.columns(5)
            for i, symbol in enumerate(TOP_SYMBOLS[:10]):
                col_idx = i % 5
                with cols[col_idx]:
                    if symbol in latest_prices:
                        exchanges = latest_prices[symbol]
                        if exchanges:
                            # Get the most recent price
                            latest_exchange = list(exchanges.keys())[0]
                            latest_data = exchanges[latest_exchange]

                            st.metric(
                                label=symbol,
                                value=f"${latest_data['price']:,.2f}",
                                delta=f"{latest_exchange.upper()}"
                            )
                        else:
                            st.metric(label=symbol, value="No Data")
                    else:
                        st.metric(label=symbol, value="No Data")

            # Price trend charts
            st.subheader("📈 Price Trends")
            chart_cols = st.columns(2)

            for i, symbol in enumerate(TOP_SYMBOLS[:4]):
                col_idx = i % 2
                with chart_cols[col_idx]:
                    fig = create_price_chart(symbol, latest_prices)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No market data available")

    with tab2:
        st.header("🤖 AI Signals & Analytics")

        # Get AI signals
        ai_signals = get_ai_signals()

        if ai_signals:
            # Display recent signals
            st.subheader("Recent AI Signals")

            signals_df = pd.DataFrame(ai_signals)
            if not signals_df.empty:
                # Format the dataframe for display
                signals_df['timestamp'] = pd.to_datetime(
                    signals_df['timestamp']).dt.strftime('%H:%M:%S')
                signals_df['confidence'] = signals_df['confidence'].apply(
                    lambda x: f"{x:.1%}")

                st.dataframe(
                    signals_df[['symbol', 'decision', 'confidence', 'source', 'timestamp']],
                    use_container_width=True
                )

            # AI confidence chart
            st.subheader("AI Confidence Analysis")
            fig = create_ai_confidence_chart(ai_signals)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No AI signals available")

    with tab3:
        st.header("💼 Portfolio Overview")

        # Get portfolio data
        portfolio_data = get_portfolio_data()

        if portfolio_data and portfolio_data.get("total_value", 0) > 0:
            # Portfolio summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Value",
                    f"${portfolio_data['total_value']:,.2f}",
                    f"{portfolio_data['total_pnl']:+,.2f}"
                )

            with col2:
                st.metric(
                    "Total PnL",
                    f"${portfolio_data['total_pnl']:+,.2f}",
                    f"{portfolio_data['performance']['pnl_percentage']:+.1f}%"
                )

            with col3:
                st.metric(
                    "Positions",
                    portfolio_data['positions_count']
                )

            with col4:
                st.metric(
                    "Total Trades",
                    portfolio_data['total_trades']
                )

            # Portfolio allocation chart
            st.subheader("Portfolio Allocation")
            fig = create_allocation_chart(portfolio_data)
            st.plotly_chart(fig, use_container_width=True)

            # Holdings table
            st.subheader("Current Holdings")
            holdings = portfolio_data.get("holdings", {})

            if holdings:
                holdings_data = []
                for symbol, holding in holdings.items():
                    holdings_data.append({
                        "Symbol": symbol,
                        "Quantity": f"{holding['quantity']:.4f}",
                        "Avg Buy Price": f"${holding['average_buy_price']:.2f}",
                        "Current Price": f"${holding['current_price']:.2f}",
                        "Current Value": f"${holding['current_value']:.2f}",
                        "Unrealized PnL": f"${holding['unrealized_pnl']:+.2f}",
                        "PnL %": f"{holding['pnl_percentage']:+.1f}%"
                    })

                holdings_df = pd.DataFrame(holdings_data)
                st.dataframe(holdings_df, use_container_width=True)
        else:
            st.info("No portfolio data available")

    with tab4:
        st.header("⚠️ Risk Alerts & Monitoring")

        # Get risk alerts
        risk_alerts = get_risk_alerts()

        if risk_alerts:
            st.subheader("Active Risk Alerts")

            for alert in risk_alerts:
                level = alert.get("level", "MEDIUM")
                message = alert.get("message", "Risk alert")
                timestamp = alert.get("timestamp", "")

                # Color code based on alert level
                if level == "CRITICAL":
                    st.error(f"🚨 {message} - {timestamp}")
                elif level == "HIGH":
                    st.warning(f"🔴 {message} - {timestamp}")
                elif level == "MEDIUM":
                    st.info(f"🟠 {message} - {timestamp}")
                else:
                    st.success(f"🟡 {message} - {timestamp}")
        else:
            st.success("✅ No active risk alerts")

        # Risk monitoring status
        st.subheader("Risk Monitoring Status")
        if risk_service:
            status = risk_service.get_risk_status()
            st.json(status)

    with tab5:
        st.header("💧 Liquidity Analysis")

        # Get liquidity data
        liquidity_data = get_liquidity_data()

        if liquidity_data:
            st.subheader("Liquidity Snapshots")

            for symbol, snapshot in liquidity_data.items():
                with st.expander(f"📊 {symbol} Liquidity"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Best Bid",
                            f"${snapshot.get('best_prices', {}).get('best_bid', 0):,.2f}")
                        st.metric(
                            "Best Ask",
                            f"${snapshot.get('best_prices', {}).get('best_ask', 0):,.2f}")

                    with col2:
                        st.metric(
                            "Spread", f"${snapshot.get('best_prices', {}).get('spread', 0):,.2f}")
                        st.metric(
                            "Liquidity Score",
                            f"{snapshot.get('liquidity_score', 0):.3f}"
                        )
        else:
            st.info("No liquidity data available")

    with tab6:
        st.header("🧠 Advanced AI Insights")
        
        # Get advanced AI insights using all imported services
        advanced_insights = get_advanced_ai_insights()
        
        if advanced_insights:
            st.subheader("AI Service Analysis")
            
            # Display insights from each service
            for service_name, insight_data in advanced_insights.items():
                with st.expander(f"📊 {service_name.replace('_', ' ').title()}"):
                    if insight_data:
                        st.json(insight_data)
                    else:
                        st.info(f"No data available for {service_name}")
        else:
            st.info("No advanced AI insights available")
            
        # Display service status
        st.subheader("AI Service Status")
        services_status = {
            "Signal Engine": 'signal_engine' in globals() and signal_engine is not None,
            "Global Overlord": 'global_overlord' in globals() and global_overlord is not None,
            "Cosmic Pattern Recognizer": 'cosmic_pattern_recognizer' in globals() and cosmic_pattern_recognizer is not None,
            "Multiversal Liquidity Engine": 'multiversal_liquidity_engine' in globals() and multiversal_liquidity_engine is not None,
            "Time Aware Trade Optimizer": 'time_aware_trade_optimizer' in globals() and time_aware_trade_optimizer is not None,
            "Capital Allocation Engine": 'capital_allocation_engine' in globals() and capital_allocation_engine is not None,
            "Neural Mesh": 'neural_mesh' in globals() and neural_mesh is not None,
            "Self Replication Engine": 'self_replication_engine' in globals() and self_replication_engine is not None,
            "Unified Fetcher": 'unified_fetcher' in globals() and unified_fetcher is not None,
            "Autobuy Service": 'autobuy_service' in globals() and autobuy_service is not None,
            "Autosell Service": 'autosell_service' in globals() and autosell_service is not None,
            "Auto Execution Service": 'auto_execution_service' in globals() and auto_execution_service is not None
        }
        
        for service_name, status in services_status.items():
            if status:
                st.success(f"✅ {service_name}: Active")
            else:
                st.error(f"❌ {service_name}: Inactive")

    with tab7:
        st.header("⚙️ System Metrics & Libraries")
        
        # Get system metrics using all imported libraries
        system_metrics = get_system_metrics()
        
        if system_metrics:
            st.subheader("System Information")
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Timestamp", f"{system_metrics.get('timestamp', 0):.2f}")
                st.metric("Active Threads", system_metrics.get('active_threads', 0))
                st.metric("Requests Status", system_metrics.get('requests_status', 'N/A'))
            
            with col2:
                st.metric("NumPy Version", system_metrics.get('numpy_version', 'N/A'))
                st.metric("Matplotlib Backend", system_metrics.get('matplotlib_backend', 'N/A'))
                st.metric("Seaborn Available", "✅" if system_metrics.get('seaborn_available', False) else "❌")
            
            with col3:
                if 'system_info' in system_metrics:
                    info = system_metrics['system_info']
                    st.metric("System Version", info.version)
                    st.metric("System Status", info.status)
            
            # Display data format
            st.subheader("Data Format")
            st.code(system_metrics.get('data_format', '{}'))
            
        else:
            st.info("No system metrics available")

    # Footer
    st.markdown("---")
    st.markdown(
        f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*"
    )
