"""
Service Initializer
Handles initialization of all services and managers
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("service_initializer")


class ServiceInitializer:
    """Manages initialization of all platform services"""

    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.service_manager: Optional[Any] = None

    def initialize_service_manager(self) -> bool:
        """Initialize the main service manager"""
        try:
            from .services.service_manager import service_manager

            self.service_manager = service_manager
            self.services["service_manager"] = service_manager
            logger.info("✅ Service manager initialized")
            return True
        except ImportError as e:
            logger.error(f"Failed to import service manager: {e}")
            return False

    def initialize_core_services(self) -> None:
        """Initialize core trading services"""
        if not self.service_manager:
            logger.warning("Service manager not available, skipping core services")
            return

        core_services = [
            ("price_fetcher", "PriceFetcher"),
            ("indicators_fetcher", "IndicatorsFetcher"),
            ("cosmic_fetcher", "CosmicFetcher"),
            ("trade_engine", "TradeEngine"),
            ("unified_signal_manager", "UnifiedSignalManager"),
            ("websocket_manager", "WebSocketManager"),
            ("shared_cache", "SharedCache"),
        ]

        for service_name, class_name in core_services:
            self._initialize_service(service_name, class_name)

    def initialize_managers(self) -> None:
        """Initialize management services"""
        managers = [
            ("auto_trading_manager", "AutoTradingManager"),
            ("signal_manager", "SignalManager"),
            ("social_trading_manager", "SocialTradingManager"),
            ("bot_manager", "BotManager"),
            ("connection_manager", "ConnectionManager"),
        ]

        for manager_name, class_name in managers:
            self._initialize_service(manager_name, class_name)

    def initialize_utility_services(self) -> None:
        """Initialize utility and monitoring services"""
        utility_services = [("notification", "NotificationService")]

        for service_name, class_name in utility_services:
            self._initialize_service(service_name, class_name)

    def initialize_ai_services(self) -> None:
        """Initialize AI and strategy services"""
        ai_services = [
            ("advanced_trading", "AdvancedTrading"),
            ("ai_strategies", "AIStrategies"),
        ]

        for service_name, class_name in ai_services:
            self._initialize_service(service_name, class_name)

    def initialize_portfolio_services(self) -> None:
        """Initialize portfolio and order services"""
        portfolio_services = [
            ("portfolio_service", "PortfolioService"),
            ("order_service", "OrderService"),
            ("signal_service", "SignalService"),
            ("analytics_service", "AnalyticsService"),
            ("market_data", "MarketDataService"),
        ]

        for service_name, class_name in portfolio_services:
            self._initialize_service(service_name, class_name)

    def _initialize_service(self, service_name: str, class_name: str) -> None:
        """Initialize a single service"""
        try:
            # Import the service class
            module = __import__(f"services.{service_name}", fromlist=[class_name])
            service_class = getattr(module, class_name)

            # Initialize with service manager if available
            if self.service_manager and hasattr(service_class, "__init__"):
                if "redis_client" in service_class.__init__.__code__.co_varnames:
                    service_instance = service_class(self.service_manager.redis_client)
                else:
                    service_instance = service_class()
            else:
                service_instance = service_class()

            self.services[service_name] = service_instance
            logger.info(f"✅ {class_name} initialized")

        except ImportError as e:
            logger.warning(f"⚠️ {class_name} not available: {e}")
        except Exception as e:
            logger.warning(f"⚠️ {class_name} initialization failed: {e}")

    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service by name"""
        return self.services.get(service_name)

    def get_all_services(self) -> Dict[str, Any]:
        """Get all initialized services"""
        return self.services.copy()


# Global service initializer instance
service_initializer = ServiceInitializer()
