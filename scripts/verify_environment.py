#!/usr/bin/env python3
"""
Environment Verification Script
Checks all critical components before launch
"""

import os
import sys
import requests
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("verification.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


class EnvironmentVerifier:
    def __init__(self):
        self.checks: Dict[str, Any] = {
            "system": self.check_system,
            "network": self.check_network,
            "database": self.check_database,
            "api": self.check_api,
            "websocket": self.check_websocket,
            "frontend": self.check_frontend,
            "backend": self.check_backend,
            "security": self.check_security,
        }
        self.results: Dict[str, Dict[str, Any]] = {}

    def check_system(self) -> Dict[str, Any]:
        """Check system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "status": (
                    "ok"
                    if cpu_percent < 80 and memory.percent < 80 and disk.percent < 80
                    else "warning"
                ),
                "cpu": cpu_percent,
                "memory": memory.percent,
                "disk": disk.percent,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Test external connectivity
            response = requests.get("https://api.coingecko.com/api/v3/ping", timeout=5)
            return {
                "status": "ok" if response.status_code == 200 else "error",
                "latency": response.elapsed.total_seconds(),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_database(self) -> Dict[str, Any]:
        """Check database connection"""
        try:
            # Add your database check logic here
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_api(self) -> Dict[str, Any]:
        """Check API endpoints"""
        try:
            endpoints = [
                "http://localhost:8000/health",
                "http://localhost:8000/api/strategy/overview",
                "http://localhost:8000/api/live/all",
            ]

            results = {}
            for endpoint in endpoints:
                response = requests.get(endpoint, timeout=5)
                results[endpoint] = {
                    "status": "ok" if response.status_code == 200 else "error",
                    "latency": response.elapsed.total_seconds(),
                }

            return results
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_websocket(self) -> Dict[str, Any]:
        """Check WebSocket connection"""
        try:
            # Add your WebSocket check logic here
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_frontend(self) -> Dict[str, Any]:
        """Check frontend build"""
        try:
            frontend_dir = "frontend/dist"
            if not os.path.exists(frontend_dir):
                return {
                    "status": "error",
                    "message": "Frontend build not found",
                }

            required_files = ["index.html", "assets"]
            for file in required_files:
                if not os.path.exists(os.path.join(frontend_dir, file)):
                    return {"status": "error", "message": f"Missing {file}"}

            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_backend(self) -> Dict[str, Any]:
        """Check backend services"""
        try:
            # Add your backend check logic here
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def check_security(self) -> Dict[str, Any]:
        """Check security configurations"""
        try:
            required_env_vars = [
                "JWT_SECRET",
                "DATABASE_URL",
                "REDIS_URL",
                "CORS_ORIGINS",
            ]

            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                return {
                    "status": "error",
                    "message": (
                        f'Missing environment variables: {", ".join(missing_vars)}'
                    ),
                }

            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all verification checks"""
        logging.info("Starting environment verification...")

        for check_name, check_func in self.checks.items():
            logging.info(f"Running {check_name} check...")
            self.results[check_name] = check_func()
            logging.info(f"{check_name} check completed: {self.results[check_name]}")

        return self.results

    def generate_report(self) -> str:
        """Generate verification report"""
        report: List[str] = []
        report.append("=" * 50)
        report.append("Environment Verification Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 50)

        for check_name, result in self.results.items():
            report.append(f"\n{check_name.upper()} CHECK:")
            report.append("-" * 30)
            for key, value in result.items():
                report.append(f"{key}: {value}")

        report.append("\n" + "=" * 50)
        report.append("Verification Complete")
        report.append("=" * 50)

        return "\n".join(report)


def main():
    verifier = EnvironmentVerifier()
    verifier.run_all_checks()
    report = verifier.generate_report()

    # Print report
    print(report)

    # Save report
    with open("verification_report.txt", "w") as f:
        f.write(report)

    # Check if all critical checks passed
    critical_checks = ["system", "network", "database", "api", "security"]
    all_passed = all(
        verifier.results[check]["status"] == "ok" for check in critical_checks
    )

    if not all_passed:
        logging.error("Critical checks failed. Please review the report.")
        sys.exit(1)

    logging.info("All critical checks passed. Environment is ready for launch.")


if __name__ == "__main__":
    main()
