#!/usr/bin/env python3
"""
Crypto Widget Startup Script
Runs both the price poller and FastAPI server
"""
import subprocess
import sys
from pathlib import Path


def start_poller():
    """Start the price poller in a separate process"""
    print("🚀 Starting price poller...")
    return subprocess.Popen(
        [sys.executable, "price_poller.py"], cwd=Path(__file__).parent
    )


def start_api_server():
    """Start the FastAPI server"""
    print("🌐 Starting FastAPI server...")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api_server:app",
            "--reload",
            "--port",
            "8000",
        ],
        cwd=Path(__file__).parent,
    )


def main():
    print("🎯 Crypto Widget System Starting...")
    print("=" * 50)

    # Start both processes
    poller_process = start_poller()
    api_process = start_api_server()

    print("\n✅ Both services started!")
    print("📊 Price poller: Running in background")
    print("🌐 API server: http://localhost:8000")
    print("📄 Widget page: Open crypto_widget/widget_page_1.html in browser")
    print("📈 Live data: http://localhost:8000/prices")
    print("\nPress Ctrl+C to stop all services...")

    try:
        # Keep running until interrupted
        poller_process.wait()
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        poller_process.terminate()
        api_process.terminate()
        print("✅ Services stopped")


if __name__ == "__main__":
    main()
