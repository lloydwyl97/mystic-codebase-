# start_trade_logging.py
"""
Startup script for the Trade Logging & Strategy Memory Engine.

This script initializes the system and provides a simple interface
to test all components.
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    import uvicorn
    from db_logger import get_active_strategies, init_db
    from mutator import run_evolution_cycle
    from reward_engine import run_daily_evaluation

    from alerts import test_discord_connection
    from dashboard import app
except ImportError as e:
    print(f"âŒ Import error: {e}")
    print("Make sure all required packages are installed:")
    print("pip install fastapi uvicorn plotly sqlalchemy")
    sys.exit(1)


def initialize_system():
    """Initialize the trade logging system"""
    print("ðŸš€ Initializing Trade Logging & Strategy Memory Engine...")

    try:
        # Initialize database
        print("ðŸ“Š Initializing database...")
        init_db()
        print("âœ… Database initialized successfully")

        # Check Discord connection
        print("ðŸ“¡ Testing Discord connection...")
        discord_ok = test_discord_connection()
        print(f"Discord: {'âœ… Connected' if discord_ok else 'âŒ Not configured'}")

        # Show initial strategies
        strategies = get_active_strategies()
        print(f"ðŸ“ˆ Found {len(strategies)} active strategies")

        print("âœ… System initialization completed!")
        return True

    except Exception as e:
        print(f"âŒ Initialization failed: {e}")
        return False


def run_dashboard():
    """Start the dashboard server"""
    print("ðŸŒ Starting dashboard server...")
    print("Dashboard will be available at: http://localhost:8080")
    print("Press Ctrl+C to stop the server")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\nðŸ›‘ Dashboard server stopped")
    except Exception as e:
        print(f"âŒ Dashboard failed to start: {e}")


def run_example():
    """Run the example trading simulation"""
    print("ðŸŽ¯ Running example trading simulation...")

    try:
        from example_usage import main

        main()
    except Exception as e:
        print(f"âŒ Example failed: {e}")


def run_evaluation():
    """Run strategy evaluation"""
    print("ðŸ“Š Running strategy evaluation...")

    try:
        results = run_daily_evaluation()
        print("âœ… Evaluation completed:")
        print(f"   Strategies evaluated: {results.get('evaluated_strategies', 0)}")
        print(f"   Strategies updated: {results.get('updated_strategies', 0)}")
        print(f"   Top performers: {len(results.get('top_performers', []))}")
        print(f"   Poor performers: {len(results.get('poor_performers', []))}")
    except Exception as e:
        print(f"âŒ Evaluation failed: {e}")


def run_evolution():
    """Run evolution cycle"""
    print("ðŸ§¬ Running evolution cycle...")

    try:
        results = run_evolution_cycle()
        print("âœ… Evolution completed:")
        print(f"   New strategies: {results.get('total_new_strategies', 0)}")
        print(f"   Mutations: {results.get('mutations_created', 0)}")
        print(f"   Crossovers: {results.get('crossovers_created', 0)}")
        print(f"   Random: {results.get('random_strategies_created', 0)}")
        print(
            f"   Active population: {results.get('population_stats', {}).get('active_strategies', 0)}"
        )
    except Exception as e:
        print(f"âŒ Evolution failed: {e}")


def show_menu():
    """Show the main menu"""
    print("\n" + "=" * 60)
    print("ðŸŽ¯ Mystic AI Trading - Trade Logging System")
    print("=" * 60)
    print("1. Initialize System")
    print("2. Start Dashboard")
    print("3. Run Example Simulation")
    print("4. Run Strategy Evaluation")
    print("5. Run Evolution Cycle")
    print("6. Show System Status")
    print("7. Exit")
    print("=" * 60)


def show_status():
    """Show system status"""
    print("\nðŸ“Š System Status:")
    print("-" * 40)

    try:
        # Check database
        strategies = get_active_strategies()
        print(f"Active strategies: {len(strategies)}")

        # Check recent trades
        from db_logger import get_recent_trades

        recent_trades = get_recent_trades(limit=10)
        print(f"Recent trades: {len(recent_trades)}")

        # Check Discord
        discord_ok = test_discord_connection()
        print(f"Discord alerts: {'âœ… Enabled' if discord_ok else 'âŒ Disabled'}")

        # Show top performers
        from reward_engine import get_top_performers

        top = get_top_performers(top_n=3, min_trades=1)
        if top:
            print(f"Top performers: {len(top)}")
            for i, performer in enumerate(top[:3], 1):
                print(f"  {i}. {performer['name']}: {performer['win_rate']:.1%} win rate")

    except Exception as e:
        print(f"âŒ Status check failed: {e}")


def main():
    """Main function"""
    print("ðŸŽ¯ Welcome to Mystic AI Trading - Trade Logging System!")

    while True:
        show_menu()

        try:
            choice = input("\nSelect an option (1-7): ").strip()

            if choice == "1":
                initialize_system()

            elif choice == "2":
                run_dashboard()

            elif choice == "3":
                run_example()

            elif choice == "4":
                run_evaluation()

            elif choice == "5":
                run_evolution()

            elif choice == "6":
                show_status()

            elif choice == "7":
                print("ðŸ‘‹ Goodbye!")
                break

            else:
                print("âŒ Invalid choice. Please select 1-7.")

        except KeyboardInterrupt:
            print("\nðŸ‘‹ Goodbye!")
            break
        except Exception as e:
            print(f"âŒ Error: {e}")


if __name__ == "__main__":
    main()


