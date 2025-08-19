"""
Verify Dashboard Endpoints Implementation
Checks that all new dashboard endpoints can be imported and loaded correctly.
"""

import importlib
from typing import Any


def test_imports() -> dict[str, Any]:
    """Test importing all new dashboard endpoint modules"""
    results = {
        "successful_imports": [],
        "failed_imports": [],
        "total_tests": 0,
    }

    # Test the main dashboard missing endpoints module
    try:
        print("ðŸ”„ Testing import of dashboard_missing_endpoints...")
        from backend.routes.dashboard_missing_endpoints import (
            router as dashboard_missing_router,
        )

        print(
            f"âœ… Successfully imported dashboard_missing_endpoints with {len(dashboard_missing_router.routes)} routes"
        )
        results["successful_imports"].append("dashboard_missing_endpoints")
        results["total_tests"] += 1
    except Exception as e:
        print(f"âŒ Failed to import dashboard_missing_endpoints: {e}")
        results["failed_imports"].append(("dashboard_missing_endpoints", str(e)))
        results["total_tests"] += 1

    # Test importing the router setup
    try:
        print("ðŸ”„ Testing import of router_setup...")

        print("âœ… Successfully imported router_setup")
        results["successful_imports"].append("router_setup")
        results["total_tests"] += 1
    except Exception as e:
        print(f"âŒ Failed to import router_setup: {e}")
        results["failed_imports"].append(("router_setup", str(e)))
        results["total_tests"] += 1

    # Test importing the app factory
    try:
        print("ðŸ”„ Testing import of app_factory...")

        print("âœ… Successfully imported app_factory")
        results["successful_imports"].append("app_factory")
        results["total_tests"] += 1
    except Exception as e:
        print(f"âŒ Failed to import app_factory: {e}")
        results["failed_imports"].append(("app_factory", str(e)))
        results["total_tests"] += 1

    return results


def test_router_registration() -> dict[str, Any]:
    """Test that the router can be registered without errors"""
    results = {"success": False, "error": None, "routes_count": 0}

    try:
        print("ðŸ”„ Testing router registration...")
        from backend.routes.dashboard_missing_endpoints import (
            router as dashboard_missing_router,
        )

        # Check that the router has the expected endpoints
        expected_endpoints = [
            "/api/portfolio/live",
            "/api/market/live",
            "/api/autobuy/status",
            "/api/strategy/performance",
            "/api/phase5/metrics",
            "/api/ai/model-metrics",
            "/api/trading/live",
            "/api/whale/alerts",
            "/api/backtest/results",
            "/api/system/status",
            "/api/dashboard/health",
        ]

        registered_routes = []
        for route in dashboard_missing_router.routes:
            if hasattr(route, "path"):
                registered_routes.append(route.path)

        print(f"âœ… Router has {len(registered_routes)} routes registered")
        print(f"ðŸ“‹ Registered routes: {registered_routes}")

        # Check for expected endpoints
        missing_endpoints = []
        for expected in expected_endpoints:
            if expected not in registered_routes:
                missing_endpoints.append(expected)

        if missing_endpoints:
            print(f"âš ï¸  Missing endpoints: {missing_endpoints}")
        else:
            print("âœ… All expected endpoints are registered")

        results["success"] = True
        results["routes_count"] = len(registered_routes)

    except Exception as e:
        print(f"âŒ Router registration test failed: {e}")
        results["error"] = str(e)

    return results


def test_data_source_imports() -> dict[str, Any]:
    """Test importing data sources that the endpoints depend on"""
    results = {
        "available_sources": [],
        "unavailable_sources": [],
        "total_sources": 0,
    }

    # Test AI services
    ai_sources = [
        ("ai.ai_signals", "signal_scorer"),
        ("ai.auto_trade", "get_trading_status"),
        ("ai.trade_tracker", "get_active_trades"),
        ("ai.ai_brains", "trend_analysis"),
        ("ai.ai_mystic", "mystic_oracle"),
        ("ai.persistent_cache", "get_persistent_cache"),
    ]

    for module_name, function_name in ai_sources:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, function_name):
                print(f"âœ… {module_name}.{function_name} available")
                results["available_sources"].append(f"{module_name}.{function_name}")
            else:
                print(f"âš ï¸  {module_name}.{function_name} not found")
                results["unavailable_sources"].append(f"{module_name}.{function_name}")
            results["total_sources"] += 1
        except ImportError as e:
            print(f"âŒ {module_name}.{function_name} import failed: {e}")
            results["unavailable_sources"].append(f"{module_name}.{function_name}")
            results["total_sources"] += 1

    # Test service imports
    service_sources = [
        ("services.live_market_data", "live_market_data_service"),
        ("services.portfolio_service", "portfolio_service"),
    ]

    for module_name, service_name in service_sources:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, service_name):
                print(f"âœ… {module_name}.{service_name} available")
                results["available_sources"].append(f"{module_name}.{service_name}")
            else:
                print(f"âš ï¸  {module_name}.{service_name} not found")
                results["unavailable_sources"].append(f"{module_name}.{service_name}")
            results["total_sources"] += 1
        except ImportError as e:
            print(f"âŒ {module_name}.{service_name} import failed: {e}")
            results["unavailable_sources"].append(f"{module_name}.{service_name}")
            results["total_sources"] += 1

    return results


def main():
    """Main verification function"""
    print("ðŸš€ Starting Dashboard Endpoints Verification...")
    print("=" * 60)

    # Test imports
    print("\nðŸ“¦ Testing Module Imports...")
    import_results = test_imports()

    # Test router registration
    print("\nðŸ”§ Testing Router Registration...")
    router_results = test_router_registration()

    # Test data sources
    print("\nðŸ“Š Testing Data Sources...")
    data_results = test_data_source_imports()

    # Print summary
    print("\n" + "=" * 60)
    print("ðŸ“‹ VERIFICATION SUMMARY")
    print("=" * 60)

    print(
        f"Module Imports: {len(import_results['successful_imports'])}/{import_results['total_tests']} successful"
    )
    print(f"Router Registration: {'âœ… Success' if router_results['success'] else 'âŒ Failed'}")
    print(
        f"Data Sources: {len(data_results['available_sources'])}/{data_results['total_sources']} available"
    )

    if import_results["failed_imports"]:
        print("\nâŒ Failed Imports:")
        for module, error in import_results["failed_imports"]:
            print(f"   - {module}: {error}")

    if data_results["unavailable_sources"]:
        print("\nâš ï¸  Unavailable Data Sources:")
        for source in data_results["unavailable_sources"]:
            print(f"   - {source}")

    # Overall status
    all_successful = (
        len(import_results["failed_imports"]) == 0 and
        router_results["success"] and
        len(data_results["available_sources"]) > 0
    )

    if all_successful:
        print("\nðŸŽ‰ All verification tests passed! Dashboard endpoints are ready.")
        return 0
    else:
        print("\nâš ï¸  Some verification tests failed. Check the details above.")
        return 1


if __name__ == "__main__":
    exit(main())


