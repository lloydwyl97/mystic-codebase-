# Mystic Trading Platform Architecture

This document outlines the architecture of the Mystic Trading Platform backend.

## Core Components

### Application Structure

- **main.py**: Main entry point for the application. Initializes FastAPI and registers routers.
- **lifespan.py**: Manages application startup and shutdown processes.
- **app_config.py**: Centralizes application configuration and setup.
- **error_handlers.py**: Provides centralized error handling for the application.
- **websocket_manager.py**: Manages WebSocket connections and provides broadcasting capabilities.

### Services

- **connection_manager.py**: Manages all external service connections with proper error handling and fallbacks.
- **enhanced_logging.py**: Provides structured logging, performance tracking, and comprehensive log management.
- **health_monitor.py**: Monitors the health of various application components.
- **services/market_data.py**: Provides market data services.

### Routes

Routes are organized by feature in the `routes` directory:

- **routes/health_routes.py**: Health check endpoints.
- **routes/websocket_routes.py**: WebSocket endpoints for real-time data.
- **routes/market_data.py**: Market data endpoints.
- **routes/binance_market_data.py**: Binance-specific market data endpoints.
- **routes/market_making.py**: Market making strategy endpoints.
- **routes/liquidity_void.py**: Liquidity void detection endpoints.
- **routes/fee_aware_positioning.py**: Fee-aware positioning strategy endpoints.
- **routes/fear_greed_orderbook.py**: Fear and greed orderbook analysis endpoints.
- **routes/spot_trading.py**: Spot trading endpoints.
- **routes/live_data.py**: Live data streaming endpoints.
- **routes/coinbase_api.py**: Coinbase API integration endpoints.

## Design Principles

1. **Modularity**: Each component has a single responsibility and is organized into its own module.
2. **Error Handling**: Comprehensive error handling with fallbacks for all external services.
3. **Configurability**: Application settings are centralized and configurable via environment variables.
4. **Observability**: Enhanced logging and monitoring for all application components.
5. **Testability**: Components are designed to be easily testable in isolation.

## Dependency Flow

```
main.py
  ├── lifespan.py
  │     ├── connection_manager.py
  │     ├── health_monitor.py
  │     └── services/market_data.py
  ├── app_config.py
  │     └── error_handlers.py
  └── routes/
        ├── health_routes.py
        ├── websocket_routes.py
        │     └── websocket_manager.py
        └── [feature]_routes.py
```

## Initialization Sequence

1. FastAPI application is created with lifespan management
2. Application is configured with middleware and settings
3. Routers are registered with the application
4. On startup, connections are initialized
5. Services are initialized with proper error handling
6. Enhanced logging is set up
7. Health monitoring is started

## Shutdown Sequence

1. Market data service is gracefully shut down
2. Health monitoring is stopped
3. All external connections are closed
4. Application shuts down gracefully
