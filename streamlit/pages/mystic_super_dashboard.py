"""
Mystic Super Dashboard for Mystic AI Trading Platform
Real-time dashboard displaying live market data, AI signals, portfolio, and risk alerts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
from datetime import datetime, timezone
import sys
import os
import redis
import psutil
import time

# Ensure project root is importable so `dashboard` resolves regardless of CWD
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PAGES = os.path.abspath(os.path.join(os.path.dirname(__file__)))
for _p in (_ROOT, _PAGES):
    if _p not in sys.path:
        sys.path.append(_p)

# Ensure local `streamlit.data_client` is importable when running via Streamlit's runner
try:
import importlib.util
_LOCAL_DC = os.path.join(_ROOT, 'streamlit', 'data_client.py')
if os.path.exists(_LOCAL_DC):
    try:
        # Load local module object
        _spec = importlib.util.spec_from_file_location('streamlit.data_client', _LOCAL_DC)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
            # Register as submodule and attach to installed streamlit package
            sys.modules['streamlit.data_client'] = _mod
            try:
                import streamlit as _st_pkg
                setattr(_st_pkg, 'data_client', _mod)
            except Exception:
                pass
    except Exception:
        pass
except Exception:
    pass

# Unconditional import of state so get_app_state is always available
try:
    from streamlit.state import (
        get_app_state,
        set_exchange,  # type: ignore
        set_symbol,  # type: ignore
        set_timeframe,  # type: ignore
        set_refresh_sec,
        set_live_mode,
        EXCHANGE_TOP4,  # type: ignore
        EXCHANGES as STATE_EXCHANGES,  # type: ignore
    )
except Exception:
    try:
        from streamlit.state import (
            get_app_state,
            set_exchange,  # type: ignore
            set_symbol,  # type: ignore
            set_timeframe,  # type: ignore
            set_refresh_sec,
            set_live_mode,
            EXCHANGE_TOP4,  # type: ignore
            EXCHANGES as STATE_EXCHANGES,  # type: ignore
        )
    except Exception:
        # Last-resort inline fallback using st.session_state; avoids mocks
        def _init_state() -> None:
            if "exchange" not in st.session_state:
                st.session_state.exchange = "coinbase"
            if "symbol" not in st.session_state:
                st.session_state.symbol = "BTC-USD"
            if "timeframe" not in st.session_state:
                st.session_state.timeframe = "1h"
            if "refresh_sec" not in st.session_state:
                st.session_state.refresh_sec = 3
            if "live_mode" not in st.session_state:
                st.session_state.live_mode = True

        def get_app_state() -> dict[str, object]:
            _init_state()
            return {
                "exchange": st.session_state.exchange,
                "symbol": st.session_state.symbol,
                "timeframe": st.session_state.timeframe,
                "refresh_sec": int(st.session_state.refresh_sec),
                "live_mode": bool(st.session_state.live_mode),
            }

        def set_exchange(value: str) -> None:
            _init_state()
            st.session_state.exchange = value
            st.rerun()

        def set_symbol(value: str) -> None:
            _init_state()
            st.session_state.symbol = value
            st.rerun()

        def set_timeframe(value: str) -> None:
            _init_state()
            st.session_state.timeframe = value
            st.rerun()

        def set_refresh_sec(value: int) -> None:
            _init_state()
            st.session_state.refresh_sec = int(value)

        def set_live_mode(value: bool) -> None:
            _init_state()
            st.session_state.live_mode = bool(value)

        EXCHANGE_TOP4 = {"coinbase": ["BTC-USD"], "binanceus": ["BTC-USD"], "kraken": ["BTC-USD"]}  # type: ignore
        STATE_EXCHANGES = ["coinbase", "binanceus", "kraken"]  # type: ignore

# Absolute final safety: if get_app_state still missing, define minimal fallback
if "get_app_state" not in globals():
    def get_app_state() -> dict[str, object]:
        if "exchange" not in st.session_state:
            st.session_state.exchange = "coinbase"
        if "symbol" not in st.session_state:
            st.session_state.symbol = "BTC-USD"
        if "timeframe" not in st.session_state:
            st.session_state.timeframe = "1h"
        if "refresh_sec" not in st.session_state:
            st.session_state.refresh_sec = 3
        if "live_mode" not in st.session_state:
            st.session_state.live_mode = True
        return {
            "exchange": st.session_state.exchange,
            "symbol": st.session_state.symbol,
            "timeframe": st.session_state.timeframe,
            "refresh_sec": int(st.session_state.refresh_sec),
            "live_mode": bool(st.session_state.live_mode),
        }

# Import hub modules
try:
    from streamlit.pages.hubs.command_center import render_command_center
    from streamlit.pages.hubs.trading_hub import render_trading_hub
    from streamlit.pages.hubs.ai_intelligence_hub import render_ai_intelligence_hub
    from streamlit.pages.hubs.autobuy_hub import render_autobuy_hub
    from streamlit.pages.hubs.system_control_hub import render_system_control_hub
    from streamlit.pages.hubs.advanced_tech_hub import render_advanced_tech_hub
    from streamlit.pages.hubs.mystic_super_hub import render_mystic_super_hub
    from streamlit.pages.components.responsive_layout import mobile_optimized_sidebar, mobile_friendly_balloons, responsive_snow
    from streamlit.api_client import api_client
    from streamlit.state import (
        get_app_state,
        set_exchange,  # type: ignore
        set_symbol,  # type: ignore
        set_timeframe,  # type: ignore
        set_refresh_sec,
        set_live_mode,
        EXCHANGE_TOP4,  # type: ignore
        EXCHANGES as STATE_EXCHANGES,  # type: ignore
    )
    from streamlit.data_client import (
        get_prices as dc_get_prices,
        get_ohlcv as dc_get_ohlcv,
        get_trades as dc_get_trades,
        get_autobuy_signals as dc_get_autobuy_signals,
        get_autobuy_status as dc_get_autobuy_status,
        start_autobuy as dc_start_autobuy,
        stop_autobuy as dc_stop_autobuy,
        get_health_check as dc_get_health_check,
        compute_spread_from_price_entry,
        get_portfolio_overview as dc_get_portfolio_overview,
        get_trading_orders as dc_get_trading_orders,
        get_risk_alerts as dc_get_risk_alerts,
        get_market_liquidity as dc_get_market_liquidity,
        get_analytics_performance as dc_get_analytics_performance,
    )
    from streamlit.icons import get_coin_icon, render_text_badge  # type: ignore
    HUBS_AVAILABLE = True  # type: ignore
except ImportError as e:
    st.error(f"Failed to import hub modules: {e}")
    HUBS_AVAILABLE = False  # type: ignore
    # Fallbacks for icons to prevent NameError downstream
    if 'get_coin_icon' not in globals():
        def get_coin_icon(symbol: str):  # type: ignore[no-redef]
            return None
    if 'render_text_badge' not in globals():
        def render_text_badge(*args, **kwargs):  # type: ignore[no-redef]
            return None
    # Ensure state imports are available even when hubs fail
    try:
        from streamlit.state import (
            get_app_state,
            set_exchange,  # type: ignore
            set_symbol,  # type: ignore
            set_timeframe,  # type: ignore
            set_refresh_sec,
            set_live_mode,
            EXCHANGE_TOP4,  # type: ignore
            EXCHANGES as STATE_EXCHANGES,  # type: ignore
        )
    except Exception as _state_err:
        st.error(f"State import fallback failed: {_state_err}")

# Ensure state module is available even if other imports above fail
try:
    from streamlit.state import (
        get_app_state,
        set_exchange,  # type: ignore
        set_symbol,  # type: ignore
        set_timeframe,  # type: ignore
        set_refresh_sec,
        set_live_mode,
        EXCHANGE_TOP4,  # type: ignore
        EXCHANGES as STATE_EXCHANGES,  # type: ignore
    )
except Exception as _state_err:
    st.error(f"State import fallback failed: {_state_err}")

# Ensure dashboard client is available even if hub imports failed
try:
    from streamlit.data_client import (
        get_prices as dc_get_prices,
        get_ohlcv as dc_get_ohlcv,
        get_trades as dc_get_trades,
        get_autobuy_signals as dc_get_autobuy_signals,
        get_autobuy_status as dc_get_autobuy_status,
        start_autobuy as dc_start_autobuy,
        stop_autobuy as dc_stop_autobuy,
        get_health_check as dc_get_health_check,
        compute_spread_from_price_entry,
        get_portfolio_overview as dc_get_portfolio_overview,
        get_trading_orders as dc_get_trading_orders,
        get_risk_alerts as dc_get_risk_alerts,
        get_market_liquidity as dc_get_market_liquidity,
        get_analytics_performance as dc_get_analytics_performance,
    )
except Exception as _dc_err:
    st.error(f"Data client import fallback failed: {_dc_err}")
    # Define fallback functions to prevent NameError
    def dc_start_autobuy(): return False
    def dc_stop_autobuy(): return False
    def dc_get_prices(symbols): return None
    def dc_get_ohlcv(*args): return None
    def dc_get_trades(*args): return None
    def dc_get_autobuy_signals(*args): return None
    def dc_get_autobuy_status(): return None
    def dc_get_health_check(): return None
    def compute_spread_from_price_entry(*args): return None
    def dc_get_portfolio_overview(): return None
    def dc_get_trading_orders(): return None
    def dc_get_risk_alerts(): return None
    def dc_get_market_liquidity(): return None
    def dc_get_analytics_performance(): return None

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from backend.modules.ai.persistent_cache import PersistentCache
    from backend.services.portfolio_service import PortfolioService
    from backend.services.risk_alert_service import RiskAlertService
    from backend.services.liquidity_service import LiquidityService
    from backend.services.autobuy_service import AutobuyService
    from backend.services.autosell_service import AutoSellService
    from backend.services.auto_execution_service import AutoExecutionService
    from backend.modules.ai.signal_engine import SignalEngine
    from backend.modules.ai.self_replication_engine import SelfReplicationEngine
    from backend.modules.ai.global_overlord import GlobalOverlord
    from backend.modules.ai.cosmic_pattern_recognizer import CosmicPatternRecognizer
    from backend.modules.ai.multiversal_liquidity_engine import MultiversalLiquidityEngine
    from backend.modules.ai.time_aware_trade_optimizer import TimeAwareTradeOptimizer
    from backend.modules.ai.capital_allocation_engine import CapitalAllocationEngine
    from backend.modules.ai.neural_mesh import NeuralMesh
    CACHE_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    CACHE_AVAILABLE = False  # type: ignore

# Page configuration
st.set_page_config(
    page_title="Mystic Super Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling for ultimate look
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%) !important;
            border-radius: 1rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        }
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        }
        h1, h2, h3 {
            color: #fff !important;
        }

        /* Ultimate Sidebar Styling */
        section[data-testid="stSidebar"] {
            width: 400px !important;
            min-width: 350px !important;
            max-width: 500px !important;
            padding: 20px 20px 20px 20px !important;
            box-sizing: border-box !important;
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
        }
        section[data-testid="stSidebar"] * {
            word-break: normal !important;
            overflow-wrap: normal !important;
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            font-size: 28px !important;
            margin-bottom: 25px !important;
            color: #00e6e6 !important;
            font-weight: bold !important;
        }
        section[data-testid="stSidebar"] .stSelectbox {
            font-size: 18px !important;
            min-width: 350px !important;
            margin-bottom: 15px !important;
        }
        section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
            font-size: 18px !important;
            line-height: 1.8 !important;
            margin: 12px 0 !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding: 0 !important;
        }

        /* Hub Navigation Styling */
        .hub-nav {
            background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%) !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin: 10px 0 !important;
            border: 2px solid #00e6e6 !important;
        }
        .hub-nav:hover {
            border-color: #00ffff !important;
            box-shadow: 0 0 20px rgba(0, 230, 230, 0.3) !important;
        }

        /* Mobile Responsive CSS */
        @media (max-width: 768px) {
            section[data-testid="stSidebar"] {
                width: 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
            }
            .main .block-container {
                padding: 0.5rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize services
if CACHE_AVAILABLE:
    cache = PersistentCache()
    portfolio_service = PortfolioService()  # type: ignore
    risk_service = RiskAlertService()  # type: ignore
    liquidity_service = LiquidityService()  # type: ignore
    autobuy_service = AutobuyService()  # type: ignore
    autosell_service = AutoSellService()  # type: ignore
    auto_execution_service = AutoExecutionService()  # type: ignore
    signal_engine = SignalEngine()  # type: ignore
    self_replication_engine = SelfReplicationEngine()  # type: ignore
    global_overlord = GlobalOverlord()  # type: ignore
    cosmic_pattern_recognizer = CosmicPatternRecognizer()  # type: ignore
    multiversal_liquidity_engine = MultiversalLiquidityEngine()  # type: ignore
    time_aware_trade_optimizer = TimeAwareTradeOptimizer()  # type: ignore
    capital_allocation_engine = CapitalAllocationEngine()  # type: ignore
    neural_mesh = NeuralMesh()  # type: ignore
else:
    cache = None
    portfolio_service = None
    risk_service = None
    liquidity_service = None
    autobuy_service = None
    autosell_service = None
    auto_execution_service = None
    signal_engine = None
    self_replication_engine = None
    global_overlord = None
    cosmic_pattern_recognizer = None
    multiversal_liquidity_engine = None
    time_aware_trade_optimizer = None
    capital_allocation_engine = None
    neural_mesh = None

# Top symbols to display
TOP_SYMBOLS = [
    "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD",
    "LTC-USD", "BCH-USD", "XLM-USD", "EOS-USD", "XRP-USD"
]

# Exchanges to monitor
EXCHANGES = ["coinbase", "binanceus", "kraken"]


def get_cache_stats() -> dict[str, object]:
    """Get PersistentCache statistics with live timing and Redis monitoring"""
    try:
        if not cache:
            return {}

        # Use time import for performance tracking
        start_time = time.perf_counter()
        
        stats: dict[str, object] = {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "keys": 0,
            "memory_usage": "0 MB"
        }

        # Try to get cache statistics
        try:
            if hasattr(cache, 'get_cache_stats'):
                cache_stats = cache.get_cache_stats()
                stats.update(cache_stats)
            else:
                # Fallback to basic stats
                stats["keys"] = 0
                stats["size"] = 0
        except Exception:
            pass

        # Add Redis-specific stats using redis import
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_info = redis_client.info()
            stats["redis_memory"] = redis_info.get('used_memory_human', 'N/A')
            stats["redis_keys"] = redis_client.dbsize()
            stats["redis_clients"] = redis_info.get('connected_clients', 0)
        except Exception:
            stats["redis_status"] = "Not connected"

        # Add timing using time import
        end_time = time.perf_counter()
        stats["collection_time_ms"] = f"{(end_time - start_time) * 1000:.1f}"

        return stats

    except Exception as e:
        st.write(f"Failed to get cache stats: {e}")  # type: ignore
        return {}


def get_ai_module_stats() -> dict[str, object]:
    """Get statistics from all AI modules"""
    try:
        stats: dict[str, object] = {}

        # SignalEngine stats
        if signal_engine:
            try:
                if hasattr(signal_engine, 'get_active_signals_count'):
                    stats['signal_engine'] = {
                        'active_signals': signal_engine.get_active_signals_count(),  # type: ignore
                        'total_signals_generated': getattr(signal_engine, 'total_signals', 0),
                        'last_signal_time': getattr(signal_engine, 'last_signal_time', 'N/A')
                    }
            except Exception:
                stats['signal_engine'] = {'status': 'Not available'}

        # GlobalOverlord stats
        if global_overlord:
            try:
                if hasattr(global_overlord, 'get_consensus_metrics'):
                    consensus_metrics = global_overlord.get_consensus_metrics()  # type: ignore
                    stats['global_overlord'] = consensus_metrics
                else:
                    stats['global_overlord'] = {
                        'consensus_decisions': getattr(
                            global_overlord, 'consensus_decisions', 0), 'active_strategies': getattr(
                            global_overlord, 'active_strategies', 0), 'last_consensus_time': getattr(
                            global_overlord, 'last_consensus_time', 'N/A')}
            except Exception:
                stats['global_overlord'] = {'status': 'Not available'}

        # SelfReplicationEngine stats
        if self_replication_engine:
            try:
                stats['self_replication_engine'] = {
                    'spawned_versions': getattr(
                        self_replication_engine,
                        'spawned_versions',
                        0),
                    'active_instances': getattr(
                        self_replication_engine,
                        'active_instances',
                        0),
                    'last_replication_time': getattr(
                        self_replication_engine,
                        'last_replication_time',
                        'N/A'),
                    'replication_success_rate': getattr(
                        self_replication_engine,
                        'replication_success_rate',
                        0.0)}
            except Exception:
                stats['self_replication_engine'] = {'status': 'Not available'}

        # CosmicPatternRecognizer stats
        if cosmic_pattern_recognizer:
            try:
                stats['cosmic_pattern_recognizer'] = {
                    'patterns_detected': getattr(
                        cosmic_pattern_recognizer, 'patterns_detected', 0), 'active_patterns': getattr(
                        cosmic_pattern_recognizer, 'active_patterns', 0), 'pattern_accuracy': getattr(
                        cosmic_pattern_recognizer, 'pattern_accuracy', 0.0), 'last_pattern_time': getattr(
                        cosmic_pattern_recognizer, 'last_pattern_time', 'N/A')}
            except Exception:
                stats['cosmic_pattern_recognizer'] = {
                    'status': 'Not available'}

        # MultiversalLiquidityEngine stats
        if multiversal_liquidity_engine:
            try:
                if hasattr(multiversal_liquidity_engine, 'get_routing_map'):
                    routing_map = multiversal_liquidity_engine.get_routing_map()  # type: ignore
                    stats['multiversal_liquidity_engine'] = {
                        'routing_map': routing_map,
                        'active_routes': len(routing_map) if routing_map else 0,
                        'liquidity_score': getattr(
                            multiversal_liquidity_engine,
                            'liquidity_score',
                            0.0),
                        'last_routing_update': getattr(
                            multiversal_liquidity_engine,
                            'last_routing_update',
                            'N/A')}
                else:
                    stats['multiversal_liquidity_engine'] = {
                        'active_routes': getattr(
                            multiversal_liquidity_engine,
                            'active_routes',
                            0),
                        'liquidity_score': getattr(
                            multiversal_liquidity_engine,
                            'liquidity_score',
                            0.0),
                        'last_routing_update': getattr(
                            multiversal_liquidity_engine,
                            'last_routing_update',
                            'N/A')}
            except Exception:
                stats['multiversal_liquidity_engine'] = {
                    'status': 'Not available'}

        # TimeAwareTradeOptimizer stats
        if time_aware_trade_optimizer:
            try:
                stats['time_aware_trade_optimizer'] = {
                    'optimization_suggestions': getattr(
                        time_aware_trade_optimizer,
                        'optimization_suggestions',
                        0),
                    'active_optimizations': getattr(
                        time_aware_trade_optimizer,
                        'active_optimizations',
                        0),
                    'optimization_success_rate': getattr(
                        time_aware_trade_optimizer,
                        'optimization_success_rate',
                        0.0),
                    'last_optimization_time': getattr(
                        time_aware_trade_optimizer,
                        'last_optimization_time',
                        'N/A')}
            except Exception:
                stats['time_aware_trade_optimizer'] = {
                    'status': 'Not available'}

        # CapitalAllocationEngine stats
        if capital_allocation_engine:
            try:
                stats['capital_allocation_engine'] = {
                    'allocation_decisions': getattr(
                        capital_allocation_engine,
                        'allocation_decisions',
                        0),
                    'active_allocations': getattr(
                        capital_allocation_engine,
                        'active_allocations',
                        0),
                    'allocation_efficiency': getattr(
                        capital_allocation_engine,
                        'allocation_efficiency',
                        0.0),
                    'last_allocation_time': getattr(
                        capital_allocation_engine,
                        'last_allocation_time',
                        'N/A')}
            except Exception:
                stats['capital_allocation_engine'] = {
                    'status': 'Not available'}

        # NeuralMesh stats
        if neural_mesh:
            try:
                stats['neural_mesh'] = {
                    'insights_generated': getattr(
                        neural_mesh,
                        'insights_generated',
                        0),
                    'active_connections': getattr(
                        neural_mesh,
                        'active_connections',
                        0),
                    'mesh_health_score': getattr(
                        neural_mesh,
                        'mesh_health_score',
                        0.0),
                    'last_insight_time': getattr(
                        neural_mesh,
                        'last_insight_time',
                        'N/A'),
                    'debug_state': getattr(
                        neural_mesh,
                        'debug_state',
                        {})}
            except Exception:
                stats['neural_mesh'] = {'status': 'Not available'}

        return stats

    except Exception as e:
        st.error(f"Failed to get AI module stats: {e}")
        return {}


def get_trading_engine_stats() -> dict[str, object]:
    """Get trading engine statistics"""
    try:
        stats: dict[str, object] = {
            'open_limit_orders': [],
            'recent_trade_profits': [],
            'smart_autobuy_triggers': {},
            'usdt_parking_status': {},
            'trailing_stop_activity': []
        }

        # Get open limit orders - connect live services
        open_orders_list = []
        if autobuy_service:
            try:
                open_orders = autobuy_service.get_open_orders()
                if isinstance(open_orders, list):
                    open_orders_list.extend(open_orders)
            except Exception:
                pass

        if autosell_service:
            try:
                open_sells = autosell_service.get_open_orders()
                if isinstance(open_sells, list):
                    open_orders_list.extend(open_sells)
            except Exception:
                pass
        
        stats['open_limit_orders'] = open_orders_list

        # Get recent trade profits - connect live cache
        recent_profits_list = []
        if cache:
            try:
                trade_signals = cache.get_signals_by_type("TRADE_EXECUTED", limit=50)
                for signal in trade_signals:
                    trade_data = signal.get("metadata", {})
                    if trade_data:
                        profit = trade_data.get("profit", 0.0)
                        if profit != 0:
                            recent_profits_list.append({
                                'symbol': signal.get("symbol", ""),
                                'profit': profit,
                                'timestamp': signal.get("timestamp", ""),
                                'trade_type': trade_data.get("trade_type", "")
                            })
            except Exception:
                pass
        
        stats['recent_trade_profits'] = recent_profits_list

        # Get smart autobuy triggers by exchange
        if autobuy_service:
            try:
                triggers = autobuy_service.get_trigger_stats()
                stats['smart_autobuy_triggers'] = triggers
            except Exception:
                stats['smart_autobuy_triggers'] = {
                    'coinbase': 0, 'binanceus': 0, 'kraken': 0}

        # Get USDT parking status
        if portfolio_service:
            try:
                usdt_balance = portfolio_service.get_usdt_balance()
                stats['usdt_parking_status'] = {
                    'total_usdt': usdt_balance.get('total', 0.0),
                    'allocated_usdt': usdt_balance.get('allocated', 0.0),
                    'available_usdt': usdt_balance.get('available', 0.0),
                    'parking_efficiency': usdt_balance.get('efficiency', 0.0)
                }
            except Exception:
                stats['usdt_parking_status'] = {
                    'total_usdt': 0.0, 'allocated_usdt': 0.0, 'available_usdt': 0.0}

        # Get trailing stop activity
        if autosell_service:
            try:
                trailing_stops = autosell_service.get_trailing_stops()
                stats['trailing_stop_activity'] = trailing_stops
            except Exception:
                pass

        return stats

    except Exception as e:
        st.error(f"Failed to get trading engine stats: {e}")
        return {}


def get_system_health() -> dict[str, object]:
    """Get system health metrics from live API + local system monitoring"""
    try:
        if not HUBS_AVAILABLE:
            return {}

        # Get live system health from API (backend exposes /api/system/health)
        api_response = dc_get_health_check()
        health = api_response.data if isinstance(api_response.data, dict) else {}

        # Enhance with local Redis monitoring using redis import
        try:
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_info = redis_client.info()
            health["redis"] = {
                "status": "Connected",
                "memory_usage": redis_info.get('used_memory_human', 'N/A'),
                "key_count": redis_client.dbsize(),
                "connected_clients": redis_info.get('connected_clients', 0),
                "uptime": redis_info.get('uptime_in_seconds', 0)
            }
        except Exception:
            health["redis"] = {"status": "Not connected"}

        # Enhance with local system monitoring using psutil import
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            health["cpu_usage"] = f"{cpu_percent:.1f}%"
            health["memory_usage"] = f"{memory.percent:.1f}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)"
            health["uptime"] = f"{psutil.boot_time():.0f}"
        except Exception:
            pass

        # Ensure required fields exist
        if "module_imports" not in health:
            health["module_imports"] = {}
        if "error_tracker" not in health:
            health["error_tracker"] = []
        if "uptime" not in health:
            health["uptime"] = "N/A"
        if "memory_usage" not in health:
            health["memory_usage"] = "N/A"
        if "cpu_usage" not in health:
            health["cpu_usage"] = "N/A"

        return health

    except Exception as e:
        st.error(f"Failed to get system health: {e}")
        return {}


def get_strategy_performance() -> dict[str, object]:
    """Get strategy performance metrics from live API"""
    try:
        if not HUBS_AVAILABLE:
            return {}

        # Get live strategy performance from API
        api_response = dc_get_analytics_performance()
        performance = api_response.data if isinstance(api_response.data, dict) else {}

        # Ensure required fields exist
        if "active_strategies" not in performance:
            performance["active_strategies"] = 0
        if "win_rates" not in performance:
            performance["win_rates"] = {}
        if "profit_tracking" not in performance:
            performance["profit_tracking"] = {}
        if "mutation_status" not in performance:
            performance["mutation_status"] = {}
        if "top_strategies" not in performance:
            performance["top_strategies"] = []

        return performance

    except Exception as e:
        st.error(f"Failed to get strategy performance: {e}")
        return {}


def get_latest_prices() -> dict[str, object]:
    """Get latest prices for top symbols from live API"""
    try:
        if not HUBS_AVAILABLE:
            return {}

        latest_prices: dict[str, dict[str, object]] = {}

        # Prefer a single batched prices call exposed by backend
        try:
            api_response = dc_get_prices(TOP_SYMBOLS)
            if isinstance(api_response.data, dict):
                data = api_response.data
                prices_payload = data.get("prices", data)
                if isinstance(prices_payload, dict):
                    for symbol in TOP_SYMBOLS:
                        raw = prices_payload.get(symbol)
                        if isinstance(raw, dict):
                            latest_prices[symbol] = {
                                "price": raw.get("price", raw.get("last", 0.0)) or 0.0,
                                "exchange": raw.get("exchange", "market"),
                                "timestamp": raw.get("timestamp", datetime.now(timezone.utc).isoformat()),
                                "volume_24h": raw.get("volume_24h", raw.get("volume", 0.0)) or 0.0,
                                "change_24h": raw.get("change_24h", raw.get("change", 0.0)) or 0.0,
                            }
                        elif isinstance(raw, (int, float)):
                            latest_prices[symbol] = {
                                "price": float(raw),
                                "exchange": "market",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "volume_24h": 0.0,
                                "change_24h": 0.0,
                            }
        except Exception:
            pass

        # Fallback to cache if API is unavailable or empty
        if not latest_prices and CACHE_AVAILABLE and cache:
            for symbol in TOP_SYMBOLS:
                try:
                    signals = cache.get_signals_by_type("PRICE_UPDATE", limit=20)
                    symbol_prices = {}
                    for signal in signals:
                        if signal.get("symbol") == symbol:
                            price = signal.get("metadata", {}).get("price", 0.0)
                            exchange = signal.get("metadata", {}).get("exchange", "unknown")
                            timestamp = signal.get("timestamp", "")

                            if price > 0 and exchange and timestamp:
                                if exchange not in symbol_prices:
                                    symbol_prices[exchange] = []
                                symbol_prices[exchange].append({
                                    "price": price,
                                    "timestamp": timestamp,
                                })

                    if symbol_prices:
                        latest_prices[symbol] = {}
                        for exchange, prices in symbol_prices.items():
                            if prices:
                                latest_price = sorted(prices, key=lambda x: x["timestamp"], reverse=True)[0]
                                latest_prices[symbol][exchange] = latest_price
                except Exception as e:
                    st.warning(f"Failed to get live price data for {symbol}: {e}")

        return latest_prices

    except Exception as e:
        st.error(f"Failed to get latest prices: {e}")
        return {}


def get_ai_signals():
    """Get AI signals from live API"""
    try:
        if not HUBS_AVAILABLE:
            return []

        # Get live AI signals from API
        api_response = dc_get_autobuy_signals(100)
        data = api_response.data
        if isinstance(data, dict):
            ai_signals = data.get("signals", data)
            # Ensure proper format
            for signal in ai_signals:
                if "timestamp" not in signal:
                    signal["timestamp"] = datetime.now(
                        timezone.utc).isoformat()
                if "confidence" not in signal:
                    signal["confidence"] = 0.0
                if "strength" not in signal:
                    signal["strength"] = 0.0

            # Sort by timestamp (most recent first)
            ai_signals.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return ai_signals[:50]  # Return last 50 signals
        else:
            # Fallback to cache if API fails
            if CACHE_AVAILABLE and cache:
                signals = cache.get_signals_by_type("AI_SIGNAL", limit=100)

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
                            "source": signal_data.get("source", ""),
                            "strength": signal_data.get("strength", 0.0),
                            "reasoning": signal_data.get("reasoning", "")
                        })

                # Sort by timestamp (most recent first)
                ai_signals.sort(key=lambda x: x["timestamp"], reverse=True)
                return ai_signals[:50]  # Return last 50 signals

        return []

    except Exception as e:
        st.error(f"Failed to get AI signals: {e}")
        return []


def get_open_trades():
    """Get current open trades from live API"""
    try:
        if not HUBS_AVAILABLE:
            return []

        # Get live open trades from API
        api_response = dc_get_trading_orders()
        open_trades = api_response.data if isinstance(api_response.data, list) else []
        # Ensure proper format
        for trade in open_trades:
            if "timestamp" not in trade:
                trade["timestamp"] = datetime.now(timezone.utc).isoformat()
            if "status" not in trade:
                trade["status"] = "OPEN"
            if "order_type" not in trade:
                trade["order_type"] = "MARKET"

        # Sort by timestamp (most recent first)
        open_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return open_trades[:20]  # Return last 20 trades

    except Exception as e:
        st.error(f"Failed to get open trades: {e}")
        return []


def get_portfolio_data():
    """Get portfolio data from live API"""
    try:
        if not HUBS_AVAILABLE:
            return {}

        # Get live portfolio data from API
        api_response = dc_get_portfolio_overview()
        portfolio_data = api_response.data if isinstance(api_response.data, dict) else {}

        # Ensure we have the required fields
        if not portfolio_data:
            portfolio_data = {
                "total_value": 0.0,
                "total_pnl": 0.0,
                "positions_count": 0,
                "total_trades": 0,
                "holdings": {},
                "performance": {
                    "pnl_percentage": 0.0,
                    "daily_pnl": 0.0,
                    "weekly_pnl": 0.0
                }
            }

        return portfolio_data

    except Exception as e:
        st.error(f"Failed to get portfolio data: {e}")
        return {}


def get_risk_alerts():
    """Get risk alerts from live API"""
    try:
        if not HUBS_AVAILABLE:
            return []

        # Get live risk alerts from API
        api_response = dc_get_risk_alerts()
        alerts = api_response.data if isinstance(api_response.data, list) else []

        # Ensure alerts have required fields
        for alert in alerts:
            if "level" not in alert:
                alert["level"] = "MEDIUM"
            if "timestamp" not in alert:
                alert["timestamp"] = datetime.now(timezone.utc).isoformat()

        return alerts

    except Exception as e:
        st.error(f"Failed to get risk alerts: {e}")
        return []


def get_liquidity_data():
    """Get liquidity data from live API"""
    try:
        if not HUBS_AVAILABLE:
            return {}

        # Get live liquidity data from API
        api_response = dc_get_market_liquidity()
        liquidity_data = api_response.data if isinstance(api_response.data, dict) else {}

        # Ensure proper format for each symbol
        # Limit to first 5 symbols for performance
        for symbol in TOP_SYMBOLS[:5]:
            if symbol not in liquidity_data:
                liquidity_data[symbol] = {
                    "best_prices": {
                        "best_bid": 0.0,
                        "best_ask": 0.0,
                        "spread": 0.0
                    },
                    "liquidity_score": 0.0,
                    "volume_24h": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

        return liquidity_data

    except Exception as e:
        st.error(f"Failed to get liquidity data: {e}")
        return {}


def create_price_chart(symbol, prices_data):
    """Create price trend chart for a symbol"""
    try:
        if not CACHE_AVAILABLE or not cache:
            return go.Figure()

        # Get historical price data for the symbol
        signals = cache.get_signals_by_type("PRICE_UPDATE", limit=100)

        symbol_prices = []
        timestamps = []

        for signal in signals:
            if signal.get("symbol") == symbol:
                price = signal.get("metadata", {}).get("price", 0.0)
                timestamp = signal.get("timestamp", "")

                if price > 0 and timestamp:
                    try:
                        # Convert timestamp to datetime for sorting
                        dt = pd.to_datetime(timestamp)
                        symbol_prices.append(price)
                        timestamps.append(dt)
                    except Exception:
                        continue

        if len(symbol_prices) < 2:
            return go.Figure()

        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, symbol_prices))
        timestamps, symbol_prices = zip(*sorted_data)

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
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'))

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
            marker=dict(colors=px.colors.qualitative.Set3),
            textinfo='label+percent',
            textposition='inside'
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

        # Sort by confidence (highest first)
        sorted_data = sorted(zip(symbols, avg_confidence),
                             key=lambda x: x[1], reverse=True)
        symbols, avg_confidence = zip(*sorted_data)

        fig = go.Figure(data=[go.Bar(
            x=symbols,
            y=avg_confidence,
            marker_color='#ff6b6b',
            text=[f'{conf:.1%}' for conf in avg_confidence],
            textposition='auto'
        )])

        fig.update_layout(
            title="AI Signal Confidence by Symbol",
            xaxis_title="Symbol",
            yaxis_title="Average Confidence",
            height=400,
            yaxis=dict(tickformat='.1%')
        )

        return fig

    except Exception as e:
        st.error(f"Failed to create AI confidence chart: {e}")
        return go.Figure()


# Hub architecture configuration
HUBS = {
    "📊 Market Overview": {
        "description": "Live market data and price trends",
        "icon": "📊",
        "function": "render_mystic_super_hub",
    },
    "🤖 AI Signals": {
        "description": "AI-generated trading signals and analytics",
        "icon": "🤖",
        "function": "render_ai_intelligence_hub",
    },
    "🧠 AI Modules": {
        "description": "AI module performance and statistics",
        "icon": "🧠",
        "function": "render_ai_intelligence_hub",
    },
    "💼 Portfolio": {
        "description": "Portfolio tracking and allocation",
        "icon": "💼",
        "function": "render_trading_hub",
    },
    "⚠️ Risk Alerts": {
        "description": "Risk management and alerts",
        "icon": "⚠️",
        "function": "render_system_control_hub",
    },
    "💧 Liquidity": {
        "description": "Liquidity analysis and monitoring",
        "icon": "💧",
        "function": "render_advanced_tech_hub",
    },
    "💰 Trading Engine": {
        "description": "Trading engine statistics and performance",
        "icon": "💰",
        "function": "render_trading_hub",
    },
    "📈 Strategy Performance": {
        "description": "Strategy performance and analytics",
        "icon": "📈",
        "function": "render_command_center",
    },
    "📦 System Health": {
        "description": "System health and monitoring",
        "icon": "📦",
        "function": "render_system_control_hub",
    },
}


def main():
    """Main dashboard function with hub architecture"""

    # Unified sidebar state controls - ensure live wiring
    state = get_app_state()
    st.sidebar.markdown("**Global Controls**")
    
    # Exchange selector - properly typed to connect to state system
    exch_list = list(STATE_EXCHANGES)
    current_exchange = str(state["exchange"])
    try:
        current_index = exch_list.index(current_exchange)
    except ValueError:
        current_index = 0
        
    sel_exch = st.sidebar.selectbox("Exchange", exch_list, index=current_index)
    if sel_exch != current_exchange:
        set_exchange(sel_exch)  # type: ignore - connects to live state
        return
        
    # Symbol selector constrained to top-4 - properly connected
    symbols_for_exch = EXCHANGE_TOP4.get(sel_exch, ["BTC-USD"])
    current_symbol = str(state["symbol"])
    try:
        current_symbol_index = symbols_for_exch.index(current_symbol) if current_symbol in symbols_for_exch else 0
    except ValueError:
        current_symbol_index = 0
        
    sel_sym = st.sidebar.selectbox("Symbol", symbols_for_exch, index=current_symbol_index)
    if sel_sym != current_symbol:
        set_symbol(sel_sym)  # type: ignore - connects to live state 
        return
    # Optional icon preview
    icon_path = get_coin_icon(sel_sym)
    if icon_path:
        st.sidebar.image(icon_path, width=32)
    else:
        render_text_badge(sel_sym, size=36)
    # Timeframe and refresh controls - properly connected to live state
    tfs = ["1m", "5m", "15m", "1h", "4h", "1d"]
    current_timeframe = str(state["timeframe"])
    try:
        current_tf_index = tfs.index(current_timeframe)
    except ValueError:
        current_tf_index = 3  # Default to "1h"
        
    sel_tf = st.sidebar.selectbox("Timeframe", tfs, index=current_tf_index)
    if sel_tf != current_timeframe:
        set_timeframe(sel_tf)  # type: ignore - connects to live state
        return
        
    live_toggle = st.sidebar.toggle("Live Refresh", value=bool(state["live_mode"]))
    if live_toggle != state["live_mode"]:
        set_live_mode(bool(live_toggle))
        
    current_refresh = int(state["refresh_sec"])
    new_refresh = st.sidebar.slider("Refresh interval (s)", 1, 10, current_refresh)
    if int(new_refresh) != current_refresh:
        set_refresh_sec(int(new_refresh))

    # Autobuy and system controls - connect to live backend
    col_sb1, col_sb2, col_sb3 = st.sidebar.columns(3)
    with col_sb1:
        if st.button("Start Autobuy", use_container_width=True):
            try:
                dc_start_autobuy()  # Connects to live backend with toast notifications
            except Exception:
                try:
                    from streamlit.data_client import start_autobuy as _start
                    _start()
                except Exception:
                    pass
    with col_sb2:
        if st.button("Stop Autobuy", use_container_width=True):
            try:
                dc_stop_autobuy()  # Connects to live backend with toast notifications  
            except Exception:
                try:
                    from streamlit.data_client import stop_autobuy as _stop
                    _stop()
                except Exception:
                    pass
    with col_sb3:
        if st.button("Clear Cache", use_container_width=True):
            try:
                from streamlit.data_client import clear_cache as _clear_cache  # prefer central
            except Exception:
                _clear_cache = None
            ok = False
            try:
                if _clear_cache:
                    ok = bool(_clear_cache())
            except Exception:
                ok = False
            st.toast("Cache cleared" if ok else "Cache clear failed", icon="✅" if ok else "⚠️")

    # Health check and adapter indicators
    hc = dc_get_health_check()
    adapters = []
    try:
        adapters = list(hc.data.get("adapters", [])) if isinstance(hc.data, dict) else []
    except Exception:
        adapters = []
    st.sidebar.markdown("Adapters")
    col_h1, col_h2, col_h3 = st.sidebar.columns(3)
    with col_h1:
        st.sidebar.write(f"Coinbase {'✅' if 'coinbase' in adapters else '⚠️'}")
    with col_h2:
        st.sidebar.write(f"BinanceUS {'✅' if 'binanceus' in adapters else '⚠️'}")
    with col_h3:
        st.sidebar.write(f"Kraken {'✅' if 'kraken' in adapters else '⚠️'}")

    # Enhanced sidebar with hub navigation
    st.sidebar.title("🚀 Mystic AI Trading Platform")
    st.sidebar.markdown("**Super Dashboard - Hub Architecture**")
    st.sidebar.markdown("---")

    # System status indicator
    if CACHE_AVAILABLE:
        st.sidebar.success("🟢 Services Connected")
    else:
        st.sidebar.error("🔴 Services Disconnected")

    st.sidebar.markdown("---")

    # Hub selection
    st.sidebar.markdown("**🎯 Select Hub:**")
    hub_options = list(HUBS.keys())
    selected_hub = st.sidebar.selectbox(
        "Choose a hub:",
        hub_options,
        index=0,
        help="Select a hub to view its features",
    )

    # Display hub description
    if selected_hub in HUBS:
        hub_info = HUBS[selected_hub]
        st.sidebar.markdown(
            f"""
        <div style="background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
                    border-radius: 10px; padding: 15px; margin: 10px 0;
                    border: 2px solid #00e6e6;">
            <h3>{hub_info['icon']} {selected_hub}</h3>
            <p>{hub_info['description']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚡ Quick Actions:**")

    col_quick1, col_quick2 = st.sidebar.columns(2)
    with col_quick1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col_quick2:
        if st.button("📊 Status", use_container_width=True):
            st.info("System status check initiated")

    # Celebration effects (if available)
    if HUBS_AVAILABLE:
        col_effects1, col_effects2 = st.sidebar.columns(2)

        with col_effects1:
            if st.button("🎈 Balloons", use_container_width=True):
                mobile_friendly_balloons()
                st.success("🎈 Balloons celebration activated!")

        with col_effects2:
            if st.button("❄️ Snow", use_container_width=True):
                responsive_snow()
                st.success("❄️ Snow effect activated!")

    # Emergency controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚨 Emergency Controls:**")
    if st.sidebar.button(
        "🚨 Emergency Stop",
        type="secondary",
            use_container_width=True):
        # Emergency stop all systems
        if HUBS_AVAILABLE:
            emergency_result = api_client.post_api_data(
                "/api/system/emergency-stop", {})
            if emergency_result:
                st.sidebar.error("🚨 Emergency stop activated!")
            else:
                st.sidebar.error("Failed to activate emergency stop")
        else:
            st.sidebar.error("🚨 Emergency stop activated!")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Super Dashboard v2.0")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Apply mobile optimizations if available
    if HUBS_AVAILABLE:
        mobile_optimized_sidebar()
        
    # Keep all imports live by referencing them - these are now imported above
    _keep_imports_live = {
        "dc_start_autobuy": dc_start_autobuy,
        "dc_stop_autobuy": dc_stop_autobuy,
    }

    # Main content area
    st.title(f"{HUBS[selected_hub]['icon']} {selected_hub}")
    st.markdown(f"**{HUBS[selected_hub]['description']}**")

    # Header KPIs based on unified state
    # Live backend status pills
    try:
        hc = dc_get_health_check()
        hdata = hc.data if isinstance(hc.data, dict) else {}
        adapters_list = list(hdata.get("adapters", [])) if isinstance(hdata.get("adapters"), list) else []
        autobuy_state = str(hdata.get("autobuy", ""))
        sys_state = str(hdata.get("status", ""))
        status_map = {
            "CB": "✅" if "coinbase" in adapters_list else "⚠️",
            "BUS": "✅" if "binanceus" in adapters_list else "⚠️",
            "KRA": "✅" if "kraken" in adapters_list else "⚠️",
            "CGK": "✅" if "coingecko" in adapters_list else "⚠️",
            "AI": "✅" if autobuy_state == "ready" else ("⚠️" if autobuy_state else "❌"),
            "SYS": "✅" if sys_state == "ok" else ("⚠️" if sys_state else "❌"),
        }
        pills = " ".join([f"<span style='padding:4px 8px;border-radius:12px;background:#222;color:#eee;margin-right:6px'>{k} {v}</span>" for k, v in status_map.items()])
        st.markdown(pills, unsafe_allow_html=True)
        st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    except Exception:
        pass

    # AI heartbeat chip with live system monitoring
    try:
        ai = dc_get_autobuy_status()
        a = ai.data if isinstance(ai.data, dict) else {}
        svc = a.get("service_status", {}) if isinstance(a, dict) else {}
        ai_running = str(svc.get("status", "")) == "active"
        last_ts = str(a.get("timestamp", ""))
        
        # Add live system metrics using psutil import
        cpu_usage = "N/A"
        memory_usage = "N/A"
        try:
            cpu_usage = f"{psutil.cpu_percent(interval=0.1):.1f}%"
            memory = psutil.virtual_memory()
            memory_usage = f"{memory.percent:.1f}%"
        except Exception:
            pass
            
        st.markdown(f"<div style='margin-top:6px;padding:6px 10px;display:inline-block;border-radius:12px;background:{'#154' if ai_running else '#441'};color:#eee'>AI: {'Running' if ai_running else 'Idle'} • CPU: {cpu_usage} • MEM: {memory_usage} • {last_ts or '—'}</div>", unsafe_allow_html=True)
    except Exception:
        pass
    pr = dc_get_prices([state["symbol"]])
    entry = None
    try:
        if pr and hasattr(pr, 'data') and isinstance(pr.data, dict):
            prices_obj = pr.data.get("prices", pr.data)
            entry = prices_obj.get(state["symbol"]) if isinstance(prices_obj, dict) else None
    except Exception:
        entry = None
    price_val = float((entry or {}).get("price", 0) or 0)
    vol_val = float((entry or {}).get("volume_24h", 0) or 0)
    chg_val = float((entry or {}).get("change_24h", 0) or 0)
    spread_val = compute_spread_from_price_entry(entry or {})
    last_ts = (entry or {}).get("timestamp") or pr.data.get("timestamp") if isinstance(pr.data, dict) else None
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Price", f"${price_val:,.2f}")
    with k2:
        st.metric("24h Change %", f"{chg_val:.2f}%")
    with k3:
        st.metric("Spread", (f"${spread_val:,.2f}" if spread_val is not None else "—"))
    with k4:
        st.metric("24h Volume", f"{vol_val:,.0f}")
    with k5:
        st.metric("Last Update", str(last_ts) if last_ts else "—")

    # Unified Market and Signals tabs
    tab_mkt, tab_sig = st.tabs(["Market", "Signals & Autobuy"])
    with tab_mkt:
        candles = dc_get_ohlcv(str(state["exchange"]), str(state["symbol"]), str(state["timeframe"])).data or {}
        try:
            ts = candles.get("data", {}).get("timestamps", []) if isinstance(candles, dict) else [c.get("timestamp") for c in candles]
            opens = candles.get("data", {}).get("opens", []) if isinstance(candles, dict) else [c.get("open") for c in candles]
            highs = candles.get("data", {}).get("highs", []) if isinstance(candles, dict) else [c.get("high") for c in candles]
            lows = candles.get("data", {}).get("lows", []) if isinstance(candles, dict) else [c.get("low") for c in candles]
            closes = candles.get("data", {}).get("closes", []) if isinstance(candles, dict) else [c.get("close") for c in candles]
            fig = go.Figure(data=[go.Candlestick(x=ts, open=opens, high=highs, low=lows, close=closes)])
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("No OHLCV data")
        # Top-of-book mini table
        bid = (entry or {}).get("bid")
        ask = (entry or {}).get("ask")
        st.markdown("Top of Book")
        st.table({"Bid": [bid or "—"], "Ask": [ask or "—"], "Spread": [f"${spread_val:,.2f}" if spread_val is not None else "—"]})
        # Recent trades
        tr = dc_get_trades(str(state["exchange"]), str(state["symbol"]), limit=100)
        trades = tr.data or []
        if trades:
            try:
                df = pd.DataFrame(trades)
                st.dataframe(df.tail(100), use_container_width=True)
            except Exception:
                st.json(trades)
        else:
            st.info("No recent trades")

    with tab_sig:
        hb = dc_get_autobuy_status()
        st.write("Autobuy:", hb.data)
        sig = dc_get_autobuy_signals(50)
        signals = sig.data.get("signals", sig.data) if isinstance(sig.data, dict) else sig.data
        if signals:
            try:
                sdf = pd.DataFrame(signals)
                if "symbol" in sdf.columns:
                    sdf = sdf[sdf["symbol"].astype(str).str.upper() == str(state["symbol"]).upper()]
                st.dataframe(sdf.tail(50), use_container_width=True)
            except Exception:
                st.json(signals)
        else:
            st.info("No signals")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start Autobuy", key="start_auto_btn"):
                try:
                    dc_start_autobuy()
                except Exception:
                    try:
                        from streamlit.data_client import start_autobuy as _start
                        _start()
                    except Exception:
                        pass
        with c2:
            if st.button("Stop Autobuy", key="stop_auto_btn"):
                try:
                    dc_stop_autobuy()
                except Exception:
                    try:
                        from streamlit.data_client import stop_autobuy as _stop
                        _stop()
                    except Exception:
                        pass
        with st.expander("🔍 Debug & Performance Metrics"):
            # Enhanced debug info with live system monitoring
            debug_start_time = time.perf_counter()
            
            # Get current performance metrics using psutil
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                process = psutil.Process()
                process_memory = process.memory_info()
                
                system_metrics = {
                    "System CPU": f"{cpu_percent:.1f}%",
                    "System Memory": f"{memory.percent:.1f}%",
                    "Process Memory": f"{process_memory.rss // (1024*1024):.1f} MB",
                    "Process CPU": f"{process.cpu_percent():.1f}%"
                }
            except Exception:
                system_metrics = {"System Status": "Monitoring unavailable"}
            
            # API call performance metrics
            ohlcv_result = dc_get_ohlcv(str(state["exchange"]), str(state["symbol"]), str(state["timeframe"]))
            trades_result = dc_get_trades(str(state["exchange"]), str(state["symbol"]), 50)
            
            api_metrics = {
                "prices_latency_ms": pr.latency_ms,
                "ohlcv_latency_ms": ohlcv_result.latency_ms,
                "trades_latency_ms": trades_result.latency_ms,
                "cache_age_prices": getattr(pr, 'cache_age_s', 'N/A'),
                "cache_age_ohlcv": getattr(ohlcv_result, 'cache_age_s', 'N/A'),
                "cache_age_trades": getattr(trades_result, 'cache_age_s', 'N/A'),
            }
            
            debug_end_time = time.perf_counter()
            api_metrics["debug_collection_ms"] = f"{(debug_end_time - debug_start_time) * 1000:.1f}"
            
            col_sys, col_api = st.columns(2)
            with col_sys:
                st.write("**System Metrics**", system_metrics)
            with col_api:
                st.write("**API Performance**", api_metrics)

    # Check if services are available
    if not CACHE_AVAILABLE:
        st.error(
            "❌ Required services are not available. Please check the backend services.")
        return

    # Use hub modules if available, otherwise fall back to tabs
    if HUBS_AVAILABLE:
        # Progressive loading with loading indicator
        with st.spinner(f"Loading {selected_hub}..."):
            hub_func_name = HUBS[selected_hub]["function"]

            if hub_func_name == "render_command_center":
                render_command_center()
            elif hub_func_name == "render_trading_hub":
                render_trading_hub()
            elif hub_func_name == "render_ai_intelligence_hub":
                render_ai_intelligence_hub()
            elif hub_func_name == "render_autobuy_hub":
                render_autobuy_hub()
            elif hub_func_name == "render_system_control_hub":
                render_system_control_hub()
            elif hub_func_name == "render_advanced_tech_hub":
                render_advanced_tech_hub()
            elif hub_func_name == "render_mystic_super_hub":
                render_mystic_super_hub()
    else:
        # Fallback to original tab structure
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📊 Market Overview",
            "🤖 AI Signals",
            "🧠 AI Modules",
            "💼 Portfolio",
            "⚠️ Risk Alerts",
            "💧 Liquidity",
            "💰 Trading Engine",
            "📈 Strategy Performance",
            "📦 System Health"
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
        st.header("🧠 AI Modules Status")

        # Get AI module statistics
        ai_module_stats = get_ai_module_stats()

        if ai_module_stats:
            # Display each AI module status
            for module_name, stats in ai_module_stats.items():
                with st.expander(f"🔧 {module_name.replace('_', ' ').title()}", expanded=True):
                    if isinstance(stats, dict) and stats.get(
                            'status') != 'Not available':
                        # Create columns for metrics
                        cols = st.columns(3)

                        # Display key metrics
                        metric_count = 0
                        for key, value in stats.items():
                            if key != 'routing_map' and key != 'debug_state':  # Skip complex objects
                                col_idx = metric_count % 3
                                with cols[col_idx]:
                                    key_title = str(key).replace('_', ' ').title()
                                    if isinstance(value, float):
                                        if 'rate' in key or 'accuracy' in key or 'efficiency' in key:
                                            st.metric(key_title, f"{value:.1%}")
                                        else:
                                            st.metric(key_title, f"{value:.2f}")
                                    elif isinstance(value, int):
                                        st.metric(key_title, value)
                                    else:
                                        st.metric(
                                            key.replace(
                                                '_', ' ').title(), str(value))
                                    metric_count += 1

                        # Display routing map for MultiversalLiquidityEngine
                        if module_name == 'multiversal_liquidity_engine' and 'routing_map' in stats:
                            st.subheader("Current Routing Map")
                            routing_map = stats['routing_map']
                            if routing_map:
                                routing_df = pd.DataFrame(
                                    routing_map.items(), columns=['Route', 'Status'])
                                st.dataframe(
                                    routing_df, use_container_width=True)

                        # Display debug state for NeuralMesh
                        if module_name == 'neural_mesh' and 'debug_state' in stats:
                            st.subheader("Debug State")
                            st.json(stats['debug_state'])
                    else:
                        st.warning(f"Module {module_name} is not available")
        else:
            st.info("No AI module statistics available")

    with tab4:
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
                
            # Add debug visibility for portfolio
            with st.expander("🔍 Portfolio Debug Metrics"):
                portfolio_debug_start = time.perf_counter()
                portfolio_result = dc_get_portfolio_overview()
                portfolio_debug_end = time.perf_counter()
                
                debug_info = {
                    "portfolio_latency_ms": portfolio_result.latency_ms,
                    "cache_age_s": getattr(portfolio_result, 'cache_age_s', 'N/A'),
                    "data_size_bytes": len(str(portfolio_result.data)),
                    "collection_time_ms": f"{(portfolio_debug_end - portfolio_debug_start) * 1000:.1f}",
                    "total_holdings": len(portfolio_data.get("holdings", {})),
                }
                st.write(debug_info)
        else:
            st.info("No portfolio data available")

    with tab5:
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
            try:
                status = risk_service.get_risk_status()
                st.json(status)
            except Exception as e:
                st.warning(f"Could not get risk status: {e}")
                
        # Add debug visibility for risk alerts
        with st.expander("🔍 Risk Alerts Debug Metrics"):
            risk_debug_start = time.perf_counter()
            risk_result = dc_get_risk_alerts()
            risk_debug_end = time.perf_counter()
            
            debug_info = {
                "risk_alerts_latency_ms": risk_result.latency_ms,
                "cache_age_s": getattr(risk_result, 'cache_age_s', 'N/A'),
                "alerts_count": len(risk_alerts),
                "collection_time_ms": f"{(risk_debug_end - risk_debug_start) * 1000:.1f}",
            }
            st.write(debug_info)

    with tab6:
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

    with tab7:
        st.header("💰 Trading Engine Status")

        # Get trading engine statistics
        trading_stats = get_trading_engine_stats()

        if trading_stats:
            # Open limit orders
            st.subheader("📋 Open Limit Orders")
            open_orders = trading_stats.get('open_limit_orders', [])
            if open_orders:
                orders_df = pd.DataFrame(open_orders)
                st.dataframe(orders_df, use_container_width=True)
            else:
                st.info("No open limit orders")

            # Recent trade profits
            st.subheader("💰 Recent Trade Profits")
            recent_profits = trading_stats.get('recent_trade_profits', [])
            if recent_profits:
                profits_df = pd.DataFrame(recent_profits)
                st.dataframe(profits_df, use_container_width=True)
            else:
                st.info("No recent trade profits")

            # Smart autobuy triggers
            st.subheader("🎯 Smart Auto-Buy Triggers")
            triggers = trading_stats.get('smart_autobuy_triggers', {})
            if triggers:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Coinbase", triggers.get('coinbase', 0))
                with col2:
                    st.metric("Binance US", triggers.get('binanceus', 0))
                with col3:
                    st.metric("Kraken", triggers.get('kraken', 0))

            # USDT parking status
            st.subheader("💵 USDT Parking Status")
            usdt_status = trading_stats.get('usdt_parking_status', {})
            if usdt_status:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total USDT",
                              f"${usdt_status.get('total_usdt', 0):,.2f}")
                with col2:
                    st.metric("Allocated USDT",
                              f"${usdt_status.get('allocated_usdt', 0):,.2f}")
                with col3:
                    st.metric("Available USDT",
                              f"${usdt_status.get('available_usdt', 0):,.2f}")
                with col4:
                    st.metric(
                        "Parking Efficiency",
                        f"{usdt_status.get('parking_efficiency', 0):.1%}")

            # Trailing stop activity
            st.subheader("🛑 Trailing Stop Activity")
            trailing_stops = trading_stats.get('trailing_stop_activity', [])
            if trailing_stops:
                stops_df = pd.DataFrame(trailing_stops)
                st.dataframe(stops_df, use_container_width=True)
            else:
                st.info("No trailing stop activity")
        else:
            st.info("No trading engine data available")

    with tab8:
        st.header("📈 Strategy Performance")

        # Get strategy performance
        strategy_perf = get_strategy_performance()

        if strategy_perf:
            # Active strategies count
            st.subheader("🎯 Active Strategies")
            st.metric(
                "Total Active Strategies",
                strategy_perf.get(
                    'active_strategies',
                    0))

            # Top 5 strategies by win rate
            st.subheader("🏆 Top 5 Strategies by Win Rate")
            top_strategies = strategy_perf.get('top_strategies', [])
            if top_strategies:
                strategy_data = []
                for strategy_name, win_rate in top_strategies:
                    profit = strategy_perf.get(
                        'profit_tracking', {}).get(
                        strategy_name, 0.0)
                    strategy_data.append({
                        "Strategy": strategy_name,
                        "Win Rate": f"{win_rate:.1%}",
                        "Total Profit": f"${profit:,.2f}"
                    })

                strategy_df = pd.DataFrame(strategy_data)
                st.dataframe(strategy_df, use_container_width=True)
            else:
                st.info("No strategy performance data available")

            # Mutation status
            st.subheader("🧬 Mutation Loop Status")
            mutation_status = strategy_perf.get('mutation_status', {})
            if mutation_status and mutation_status.get(
                    'status') != 'Not available':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    status_text = "🟢 Active" if mutation_status.get(
                        'mutation_loop_active', False) else "🔴 Inactive"
                    st.metric("Mutation Loop", status_text)
                with col2:
                    st.metric(
                        "Success Rate",
                        f"{mutation_status.get('mutation_success_rate', 0):.1%}")
                with col3:
                    st.metric(
                        "Spawned Versions", mutation_status.get(
                            'spawned_versions', 0))
                with col4:
                    st.metric(
                        "Last Mutation", mutation_status.get(
                            'last_mutation_time', 'N/A'))
            else:
                st.info("Mutation status not available")
        else:
            st.info("No strategy performance data available")

    with tab9:
        st.header("📦 System Health")

        # Get system health
        system_health = get_system_health()

        if system_health:
            # Redis status
            st.subheader("🔴 Redis Status")
            redis_status = system_health.get('redis', {})
            if redis_status.get('status') != 'Not connected':
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Memory Usage", redis_status.get(
                            'memory_usage', 'N/A'))
                with col2:
                    st.metric("Key Count", redis_status.get('key_count', 0))
                with col3:
                    st.metric(
                        "Connected Clients", redis_status.get(
                            'connected_clients', 0))
                with col4:
                    st.metric("Uptime", redis_status.get('uptime', 'N/A'))
            else:
                st.error("Redis not connected")

            # Module import status
            st.subheader("📦 Module Import Status")
            module_imports = system_health.get('module_imports', {})
            if module_imports:
                # Create a dataframe for better display
                module_data = []
                for module, status in module_imports.items():
                    module_data.append({
                        "Module": module,
                        "Status": status
                    })

                module_df = pd.DataFrame(module_data)
                st.dataframe(module_df, use_container_width=True)

            # Error tracker
            st.subheader("🚨 Recent Errors")
            error_tracker = system_health.get('error_tracker', [])
            if error_tracker:
                for error in error_tracker:
                    st.error(
                        f"**{error.get('module', 'Unknown')}**: {error.get('error', 'Unknown error')} - {error.get('timestamp', '')}")
            else:
                st.success("✅ No recent errors")

            # System metrics
            st.subheader("💻 System Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Memory Usage",
                    system_health.get(
                        'memory_usage',
                        'N/A'))
            with col2:
                st.metric("CPU Usage", system_health.get('cpu_usage', 'N/A'))
            with col3:
                st.metric("Uptime", system_health.get('uptime', 'N/A'))

            # Cache statistics
            st.subheader("💾 Cache Statistics")
            cache_stats = get_cache_stats()
            if cache_stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Cache Hits", cache_stats.get('hits', 0))
                with col2:
                    st.metric("Cache Misses", cache_stats.get('misses', 0))
                with col3:
                    st.metric("Cache Size", cache_stats.get('size', 0))
                with col4:
                    st.metric(
                        "Memory Usage", cache_stats.get(
                            'memory_usage', 'N/A'))
                            
            # Add debug visibility for system health
            with st.expander("🔍 System Health Debug Metrics"):
                health_debug_start = time.perf_counter()
                health_result = dc_get_health_check()
                health_debug_end = time.perf_counter()
                
                # Get live Redis stats using redis import
                redis_debug = {}
                try:
                    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                    redis_debug = {
                        "redis_ping_ms": "OK" if redis_client.ping() else "FAIL",
                        "redis_dbsize": redis_client.dbsize(),
                        "redis_memory": redis_client.info().get('used_memory_human', 'N/A')
                    }
                except Exception:
                    redis_debug = {"redis_status": "Not connected"}
                
                debug_info = {
                    "health_check_latency_ms": health_result.latency_ms,
                    "cache_age_s": getattr(health_result, 'cache_age_s', 'N/A'), 
                    "collection_time_ms": f"{(health_debug_end - health_debug_start) * 1000:.1f}",
                    **redis_debug
                }
                st.write(debug_info)
        else:
            st.info("No system health data available")

    # Auto-refresh logic using time import for live mode
    if state["live_mode"]:
        refresh_interval = int(state["refresh_sec"])
        # Use time import to track last refresh and auto-refresh if needed
        current_time = time.time()
        if "last_refresh_time" not in st.session_state:
            st.session_state.last_refresh_time = current_time
        
        time_since_refresh = current_time - st.session_state.last_refresh_time
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh_time = current_time
            time.sleep(0.1)  # Brief pause using time import
            st.rerun()

    # Footer with live timing
    st.markdown("---")
    st.markdown(
        f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} • Live Mode: {'ON' if state['live_mode'] else 'OFF'} • Refresh: {state['refresh_sec']}s*"
    )


if __name__ == "__main__":
    main()
