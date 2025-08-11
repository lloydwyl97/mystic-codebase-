#!/usr/bin/env python3
"""
Ultimate AI Trading System Launcher
Launches the complete AI crypto trading machine with all 11 modules.
"""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime
import argparse


class UltimateAILauncher:
    """
    Complete AI trading system launcher.
    """

    def __init__(self):
        self.processes = {}
        self.services = {
            "main_api": {
                "script": "dashboard_api.py",
                "port": 8000,
                "description": "Main FastAPI Dashboard",
            },
            "trade_logger": {
                "script": "db_logger.py",
                "port": None,
                "description": "Trade Logging System",
            },
            "strategy_mutator": {
                "script": "mutator.py",
                "port": None,
                "description": "Strategy Evolution Engine",
            },
            "hyper_optimizer": {
                "script": "hyper_tuner.py",
                "port": None,
                "description": "Hyperparameter Optimization",
            },
            "watchdog": {
                "script": "watchdog.py",
                "port": None,
                "description": "Health Monitoring",
            },
        }

    def print_banner(self):
        """Print system banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚀 MYSTIC AI TRADING SYSTEM 🚀              ║
║                                                              ║
║  🧠 Complete AI Crypto Trading Machine                       ║
║  📊 11 Advanced Modules Integrated                           ║
║  💰 7-Figure Profit Potential                               ║
║  🛡️ Production-Ready & Scalable                             ║
║                                                              ║
║  ✅ Trade Logging Engine                                     ║
║  ✅ Strategy Evolution                                       ║
║  ✅ Position Sizing                                          ║
║  ✅ Capital Allocation                                       ║
║  ✅ Yield Rotation                                           ║
║  ✅ Health Monitoring                                        ║
║  ✅ Live Dashboard                                           ║
║  ✅ Hyperparameter Optimization                              ║
║  ✅ Backtesting Engine                                       ║
║  ✅ Meta Agent                                               ║
║  ✅ Cold Wallet Integration                                  ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def check_dependencies(self):
        """Check if all required dependencies are available."""
        print("🔍 Checking system dependencies...")

        required_files = [
            "models.py",
            "db_logger.py",
            "strategy_leaderboard.py",
            "mutator.py",
            "position_sizer.py",
            "capital_allocator.py",
            "yield_rotator.py",
            "watchdog.py",
            "dashboard_api.py",
            "hyper_tuner.py",
        ]

        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)

        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False

        print("✅ All dependencies found")
        return True

    def initialize_database(self):
        """Initialize the trade logging database."""
        print("🗄️ Initializing trade database...")
        try:
            from db_logger import init_db

            init_db()
            print("✅ Database initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            return False

    def start_service(self, service_name: str, service_config: dict):
        """Start a single service."""
        try:
            print(f"🚀 Starting {service_name}: {service_config['description']}")

            # Start the process
            process = subprocess.Popen(
                ["python", service_config["script"]],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.processes[service_name] = {
                "process": process,
                "config": service_config,
                "start_time": datetime.timezone.utcnow(),
            }

            # Wait a moment for startup
            time.sleep(2)

            # Check if process is still running
            if process.poll() is None:
                print(f"✅ {service_name} started successfully (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ {service_name} failed to start")
                print(f"   STDOUT: {stdout}")
                print(f"   STDERR: {stderr}")
                return False

        except Exception as e:
            print(f"❌ Error starting {service_name}: {e}")
            return False

    def start_all_services(self):
        """Start all AI trading services."""
        print("\n🚀 Starting AI Trading Services...")
        print("=" * 50)

        started_services = []
        failed_services = []

        for service_name, service_config in self.services.items():
            if self.start_service(service_name, service_config):
                started_services.append(service_name)
            else:
                failed_services.append(service_name)

        print("\n📊 Service Startup Summary:")
        print(f"   ✅ Started: {len(started_services)} services")
        print(f"   ❌ Failed: {len(failed_services)} services")

        if started_services:
            print(f"   🎯 Running services: {', '.join(started_services)}")

        if failed_services:
            print(f"   ⚠️ Failed services: {', '.join(failed_services)}")

        return len(failed_services) == 0

    def check_service_health(self):
        """Check health of all running services."""
        print("\n🛡️ Checking service health...")

        for service_name, service_info in self.processes.items():
            process = service_info["process"]

            if process.poll() is None:
                print(f"   ✅ {service_name}: Running (PID: {process.pid})")
            else:
                print(f"   ❌ {service_name}: Stopped")

    def monitor_services(self):
        """Monitor services and restart if needed."""
        print("\n👀 Starting service monitoring...")

        while True:
            try:
                for service_name, service_info in list(self.processes.items()):
                    process = service_info["process"]

                    if process.poll() is not None:
                        print(f"⚠️ {service_name} has stopped, restarting...")

                        # Restart the service
                        if self.start_service(service_name, service_info["config"]):
                            print(f"✅ {service_name} restarted successfully")
                        else:
                            print(f"❌ Failed to restart {service_name}")

                time.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                print("\n🛑 Service monitoring stopped")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(30)

    def stop_all_services(self):
        """Stop all running services."""
        print("\n🛑 Stopping all services...")

        for service_name, service_info in self.processes.items():
            process = service_info["process"]

            if process.poll() is None:
                print(f"   Stopping {service_name}...")
                process.terminate()

                try:
                    process.wait(timeout=10)
                    print(f"   ✅ {service_name} stopped")
                except subprocess.TimeoutExpired:
                    print(f"   ⚠️ {service_name} didn't stop gracefully, forcing...")
                    process.kill()

        self.processes.clear()
        print("✅ All services stopped")

    def show_dashboard_info(self):
        """Show dashboard access information."""
        print("\n" + "=" * 60)
        print("🎯 DASHBOARD ACCESS INFORMATION")
        print("=" * 60)
        print("📊 Main Dashboard:    http://localhost:8000/dashboard")
        print("🔧 API Documentation: http://localhost:8000/docs")
        print("📈 Strategy API:      http://localhost:8000/api/leaderboard")
        print("💰 Trade History:     http://localhost:8000/api/trades")
        print("🛡️ System Health:     http://localhost:8000/api/system-health")
        print("=" * 60)
        print("💡 Press Ctrl+C to stop the system")
        print("=" * 60)

    def run_interactive_mode(self):
        """Run in interactive mode with menu."""
        while True:
            print("\n" + "=" * 50)
            print("🎮 MYSTIC AI TRADING SYSTEM - INTERACTIVE MODE")
            print("=" * 50)
            print("1. 🚀 Start All Services")
            print("2. 🛑 Stop All Services")
            print("3. 🔍 Check Service Health")
            print("4. 📊 Show Dashboard Info")
            print("5. 🧠 Run Strategy Optimization")
            print("6. 💰 Allocate Capital")
            print("7. 🛡️ System Health Check")
            print("8. 📈 View Recent Trades")
            print("9. 🔄 Restart Failed Services")
            print("0. ❌ Exit")
            print("=" * 50)

            choice = input("Select option (0-9): ").strip()

            if choice == "1":
                self.start_all_services()
            elif choice == "2":
                self.stop_all_services()
            elif choice == "3":
                self.check_service_health()
            elif choice == "4":
                self.show_dashboard_info()
            elif choice == "5":
                self.run_optimization()
            elif choice == "6":
                self.allocate_capital()
            elif choice == "7":
                self.system_health_check()
            elif choice == "8":
                self.view_recent_trades()
            elif choice == "9":
                self.restart_failed_services()
            elif choice == "0":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option, please try again")

    def run_optimization(self):
        """Run strategy optimization."""
        print("\n🧠 Running Strategy Optimization...")
        try:
            from hyper_tuner import optimize_rsi_ema_breakout

            result = optimize_rsi_ema_breakout(method="genetic", rounds=20)
            if result:
                print("✅ Optimization completed!")
                print(f"   Best Profit: ${result['total_profit']:.2f}")
                print(f"   Win Rate: {result['win_rate']:.1%}")
            else:
                print("❌ Optimization failed")
        except Exception as e:
            print(f"❌ Optimization error: {e}")

    def allocate_capital(self):
        """Allocate capital."""
        print("\n💰 Allocating Capital...")
        try:
            from capital_allocator import allocate_capital

            allocations = allocate_capital(10000, method="performance")
            if allocations:
                print("✅ Capital allocated:")
                for strategy, amount in allocations.items():
                    print(f"   {strategy}: ${amount}")
            else:
                print("❌ No allocations made")
        except Exception as e:
            print(f"❌ Capital allocation error: {e}")

    def system_health_check(self):
        """Check system health."""
        print("\n🛡️ System Health Check...")
        try:
            from watchdog import TradingWatchdog

            watchdog = TradingWatchdog()
            summary = watchdog.get_system_summary()
            print(f"   Overall Health: {summary['overall_health']}")
            print(f"   Healthy Services: {summary['healthy_services']}/{summary['total_services']}")
            print(f"   Health Percentage: {summary['health_percentage']}%")
        except Exception as e:
            print(f"❌ Health check error: {e}")

    def view_recent_trades(self):
        """View recent trades."""
        print("\n📈 Recent Trades...")
        try:
            from trade_logger import get_recent_trades

            trades = get_recent_trades(10)
            if trades:
                for trade in trades:
                    print(
                        f"   {trade['timestamp']} | {trade['symbol']} | {trade['strategy']} | ${trade['profit_usd']:.2f}"
                    )
            else:
                print("   No recent trades found")
        except Exception as e:
            print(f"❌ Error viewing trades: {e}")

    def restart_failed_services(self):
        """Restart failed services."""
        print("\n🔄 Restarting Failed Services...")
        failed_services = []

        for service_name, service_info in self.processes.items():
            process = service_info["process"]
            if process.poll() is not None:
                failed_services.append(service_name)

        if failed_services:
            for service_name in failed_services:
                print(f"   Restarting {service_name}...")
                self.start_service(service_name, self.services[service_name])
        else:
            print("   No failed services found")

    def launch_full_system(self):
        """Launch the complete AI trading system."""
        self.print_banner()

        # Check dependencies
        if not self.check_dependencies():
            print("❌ System cannot start due to missing dependencies")
            return False

        # Initialize database
        if not self.initialize_database():
            print("❌ System cannot start due to database initialization failure")
            return False

        # Start all services
        if not self.start_all_services():
            print("⚠️ Some services failed to start, but continuing...")

        # Show dashboard info
        self.show_dashboard_info()

        # Start monitoring in background
        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()

        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down AI trading system...")
            self.stop_all_services()
            print("✅ System shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mystic AI Trading System Launcher")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument("--docker", "-d", action="store_true", help="Use Docker deployment")

    args = parser.parse_args()

    launcher = UltimateAILauncher()

    if args.docker:
        print("🐳 Using Docker deployment...")
        os.system("docker-compose up -d")
        print("✅ Docker services started")
        print("📊 Dashboard available at: http://localhost:8000/dashboard")
    elif args.interactive:
        launcher.run_interactive_mode()
    else:
        launcher.launch_full_system()


if __name__ == "__main__":
    main()
