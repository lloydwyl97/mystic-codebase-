# Router Organization Guide

## Current Improvements Made

### ✅ Completed Cleanup

1. **Removed Duplicate Route Declarations**
   - Removed duplicate WebSocket endpoints (`/ws/market-data` vs `/api/ws/market-data`)
   - Removed duplicate social leaderboard endpoints (`/social/leaderboard` vs `/api/social/leaderboard`)
   - Removed duplicate mobile endpoints (`/mobile/` vs `/api/mobile/`)

2. **Organized Mock Classes**
   - Created `mocks.py` file to centralize all mock implementations
   - Moved `MockSignalManager`, `MockAutoTradingManager`, `MockNotificationService`, `MockRedisClient`, `MockHealthMonitor` to separate file
   - Updated imports to use the centralized mock classes

3. **Cleaned Up Imports**
   - Removed unused imports (`BackgroundTasks`, `FastAPI`, `JSONResponse`, `BinanceAPI`, `CoinbaseAPI`, `TradingSignal`, `SignalType`, `notification_manager`)
   - Organized imports into logical groups

## 🔄 Suggested Router Splitting

The current `api_endpoints.py` file is quite large (1900+ lines). Consider splitting it into logical routers:

### Proposed Router Structure

```bash
backend/routes/
├── __init__.py
├── analytics_router.py      # Analytics & performance endpoints
├── auth_router.py          # Authentication endpoints
├── auto_trading_router.py  # Auto-trading & bot endpoints
├── exchange_router.py      # Exchange integration endpoints
├── market_router.py        # Market data & coin state endpoints
├── mobile_router.py        # Mobile & PWA endpoints
├── notification_router.py  # Notification endpoints
├── order_router.py         # Order management endpoints
├── portfolio_router.py     # Portfolio management endpoints
├── signal_router.py        # Signal management endpoints
├── social_router.py        # Social trading endpoints
├── websocket_router.py     # WebSocket endpoints
└── health_router.py        # Health & monitoring endpoints
```

### Example Router Split

**analytics_router.py:**

```python
from fastapi import APIRouter, HTTPException, Depends
from services.analytics_service import analytics_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("/performance")
async def get_performance_metrics(timeframe: str = "30d"):
    # Implementation...

@router.get("/trade-history")
async def get_trade_history(limit: int = 100, offset: int = 0):
    # Implementation...

@router.get("/strategies")
async def get_strategy_performance():
    # Implementation...
```

**Main router setup:**

```python
# main.py or router_setup.py
from fastapi import APIRouter
from routes import (
    analytics_router, auth_router, auto_trading_router,
    exchange_router, market_router, mobile_router,
    notification_router, order_router, portfolio_router,
    signal_router, social_router, websocket_router, health_router
)

app = FastAPI()

# Include all routers
app.include_router(analytics_router.router)
app.include_router(auth_router.router)
app.include_router(auto_trading_router.router)
# ... etc
```

## 🎯 Benefits of Router Splitting

1. **Maintainability:** Easier to find and modify specific endpoint groups
2. **Team Collaboration:** Different developers can work on different routers
3. **Testing:** Easier to write focused tests for specific functionality
4. **Documentation:** Better organization in OpenAPI/Swagger docs
5. **Code Reuse:** Routers can be reused in different applications

## 🔧 Implementation Steps

1. Create `backend/routes/` directory
2. Create individual router files with appropriate prefixes and tags
3. Move endpoints to their respective routers
4. Update main application to include all routers
5. Update imports and dependencies
6. Test all endpoints to ensure they still work

## 📝 Current Endpoint Categories

Based on the current `api_endpoints.py`, here's how endpoints should be distributed:

- **Analytics Router:** `/api/analytics/*` (performance, trade-history, strategies, ai-insights, portfolio-performance, risk-metrics, market-analysis)
- **Auth Router:** `/api/auth/*` (login, logout, refresh)
- **Auto Trading Router:** `/api/auto-trade/*`, `/api/auto-bot/*`
- **Exchange Router:** `/exchanges/*`
- **Market Router:** `/api/coinstate`, `/api/buy/*`
- **Mobile Router:** `/api/mobile/*` (push-subscription, background-sync, offline-data)
- **Notification Router:** `/api/notifications/*`
- **Order Router:** `/api/orders/*`
- **Portfolio Router:** `/api/portfolio/*`
- **Signal Router:** `/api/signals/*`
- **Social Router:** `/api/social/*`, `/social/*`
- **WebSocket Router:** `/api/ws/*`
- **Health Router:** `/health`, `/api/health/*`

## 🚀 Next Steps

1. **Immediate:** The current cleanup is sufficient for now
2. **Future:** Consider implementing router splitting when the codebase grows larger
3. **Optional:** Add automatic duplicate endpoint detection in CI/CD pipeline
