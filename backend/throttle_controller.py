#!/usr/bin/env python3
"""
Throttle Controller for Mystic Trading Platform

Simple interface to control API throttling and monitor performance.
Usage: python throttle_controller.py [command] [options]
"""

import argparse
import sys
import time
from typing import Any


def get_performance_dashboard() -> dict[str, Any]:
    """Get current performance dashboard"""
    try:
        from performance_monitor import performance_monitor

        return performance_monitor.get_performance_dashboard()
    except ImportError:
        print("âŒ Performance monitor not available")
        return {}


def get_api_stats() -> dict[str, Any]:
    """Get API throttling statistics"""
    try:
        from api_throttler import api_throttler

        return api_throttler.get_performance_stats()
    except ImportError:
        print("âŒ API throttler not available")
        return {}


def increase_throttling():
    """Increase API throttling level"""
    try:
        from api_throttler import api_throttler

        api_throttler.increase_throttling()
        print("âœ… Throttling increased")
    except ImportError:
        print("âŒ API throttler not available")


def decrease_throttling():
    """Decrease API throttling level"""
    try:
        from api_throttler import api_throttler

        api_throttler.decrease_throttling()
        print("âœ… Throttling decreased")
    except ImportError:
        print("âŒ API throttler not available")


def optimize_system():
    """Run automatic system optimization"""
    try:
        from performance_monitor import performance_monitor

        performance_monitor.optimize_system()
        print("âœ… System optimization completed")
    except ImportError:
        print("âŒ Performance monitor not available")


def clear_caches():
    """Clear all caches"""
    try:
        from database_optimized import optimized_db_manager
        from optimized_market_data import optimized_market_service

        optimized_market_service.clear_cache()
        optimized_db_manager.clear_cache()
        print("âœ… All caches cleared")
    except ImportError:
        print("âŒ Optimized services not available")


def show_status():
    """Show current system status"""
    print("\nðŸ” SYSTEM STATUS")
    print("=" * 50)

    # Get performance dashboard
    dashboard = get_performance_dashboard()
    if dashboard:
        health = dashboard.get("system_health", {})
        print(f"Overall Health: {health.get('overall', 'unknown')}")
        print(f"Database: {'âœ…' if health.get('database') else 'âŒ'}")
        print(f"API: {'âœ…' if health.get('api') else 'âŒ'}")
        print(f"Cache: {'âœ…' if health.get('cache') else 'âŒ'}")

        if health.get("issues"):
            print(f"Issues: {', '.join(health['issues'])}")

    # Get API stats
    api_stats = get_api_stats()
    if api_stats:
        print("\nðŸ“Š API STATISTICS")
        print(f"Total Requests: {api_stats.get('total_requests', 0)}")
        print(f"Throttled Requests: {api_stats.get('throttled_requests', 0)}")
        print(f"Success Rate: {api_stats.get('success_rate', 0):.2%}")
        print(f"Average Response Time: {api_stats.get('average_response_time', 0):.3f}s")
        print(f"Current Level: {api_stats.get('current_throttle_level', 'unknown')}")


def show_recommendations():
    """Show optimization recommendations"""
    print("\nðŸ’¡ OPTIMIZATION RECOMMENDATIONS")
    print("=" * 50)

    dashboard = get_performance_dashboard()
    if dashboard:
        recommendations = dashboard.get("optimization_recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No recommendations at this time.")


def monitor_performance(duration: int = 60):
    """Monitor performance for specified duration"""
    print(f"\nðŸ“Š MONITORING PERFORMANCE FOR {duration} SECONDS")
    print("=" * 50)
    print("Press Ctrl+C to stop early")

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            dashboard = get_performance_dashboard()
            if dashboard:
                health = dashboard.get("system_health", {})
                print(
                    f"\r[{time.strftime('%H:%M:%S')}] Health: {health.get('overall', 'unknown')} | "
                    f"API: {'âœ…' if health.get('api') else 'âŒ'} | "
                    f"DB: {'âœ…' if health.get('database') else 'âŒ'} | "
                    f"Cache: {'âœ…' if health.get('cache') else 'âŒ'}",
                    end="",
                )

            time.sleep(5)

        print("\nâœ… Monitoring completed")

    except KeyboardInterrupt:
        print("\nâ¹ï¸  Monitoring stopped by user")


def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(description="Mystic Trading Platform Throttle Controller")
    parser.add_argument(
        "command",
        choices=[
            "status",
            "increase",
            "decrease",
            "optimize",
            "clear",
            "recommendations",
            "monitor",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration for monitoring (default: 60 seconds)",
    )

    args = parser.parse_args()

    if args.command == "status":
        show_status()
    elif args.command == "increase":
        increase_throttling()
    elif args.command == "decrease":
        decrease_throttling()
    elif args.command == "optimize":
        optimize_system()
    elif args.command == "clear":
        clear_caches()
    elif args.command == "recommendations":
        show_recommendations()
    elif args.command == "monitor":
        monitor_performance(args.duration)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("ðŸš€ Mystic Trading Platform Throttle Controller")
        print("=" * 50)
        print("Available commands:")
        print("  status          - Show current system status")
        print("  increase        - Increase API throttling")
        print("  decrease        - Decrease API throttling")
        print("  optimize        - Run automatic optimization")
        print("  clear           - Clear all caches")
        print("  recommendations - Show optimization recommendations")
        print("  monitor         - Monitor performance in real-time")
        print("\nUsage: python throttle_controller.py [command]")
        print("Example: python throttle_controller.py status")
    else:
        main()


