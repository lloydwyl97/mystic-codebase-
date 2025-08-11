#!/usr/bin/env python3
"""
Endpoint Health Check Script
Tests all endpoints and generates a comprehensive report for Docker deployment
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
import httpx
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndpointHealthChecker:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": base_url,
            "summary": {
                "total_endpoints": 0,
                "successful": 0,
                "failed": 0,
                "errors": 0,
            },
            "endpoints": [],
        }

    async def test_endpoint(
        self, endpoint: str, method: str = "GET", expected_status: int = 200
    ) -> Dict[str, Any]:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method.upper() == "GET":
                    response = await client.get(url)
                elif method.upper() == "POST":
                    response = await client.post(url)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response_time = time.time() - start_time

                result = {
                    "endpoint": endpoint,
                    "method": method,
                    "url": url,
                    "status_code": response.status_code,
                    "response_time": round(response_time, 3),
                    "success": response.status_code == expected_status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if response.status_code == expected_status:
                    result["status"] = "SUCCESS"
                    self.results["summary"]["successful"] += 1
                else:
                    result["status"] = "FAILED"
                    result["error"] = f"Expected {expected_status}, got {response.status_code}"
                    self.results["summary"]["failed"] += 1

                return result

        except Exception as e:
            response_time = time.time() - start_time
            result = {
                "endpoint": endpoint,
                "method": method,
                "url": url,
                "status_code": None,
                "response_time": round(response_time, 3),
                "success": False,
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.results["summary"]["errors"] += 1
            return result

    def get_all_endpoints(self) -> List[Dict[str, str]]:
        """Define all endpoints to test"""
        return [
            # Basic health endpoints
            {"endpoint": "/health", "method": "GET", "expected_status": 200},
            {
                "endpoint": "/api/health",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/version",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/favicon.ico",
                "method": "GET",
                "expected_status": 200,
            },
            # Live market data endpoints
            {
                "endpoint": "/api/live/market-data",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/live/price/bitcoin",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/live/price/ethereum",
                "method": "GET",
                "expected_status": 200,
            },
            # Live trading endpoints
            {
                "endpoint": "/api/live/trading/signals",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/live/orders",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/live/trades/history",
                "method": "GET",
                "expected_status": 200,
            },
            # Live portfolio endpoints
            {
                "endpoint": "/api/live/portfolio/positions",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/live/portfolio/summary",
                "method": "GET",
                "expected_status": 200,
            },
            # Live AI endpoints
            {
                "endpoint": "/api/live/ai/predictions",
                "method": "GET",
                "expected_status": 200,
            },
            # Live social endpoints
            {
                "endpoint": "/api/live/social/leaderboard",
                "method": "GET",
                "expected_status": 200,
            },
            # Live bot endpoints
            {
                "endpoint": "/api/live/bots/status",
                "method": "GET",
                "expected_status": 200,
            },
            # Live notification endpoints
            {
                "endpoint": "/api/live/notifications",
                "method": "GET",
                "expected_status": 200,
            },
            # Live analytics endpoints
            {
                "endpoint": "/api/live/analytics/performance",
                "method": "GET",
                "expected_status": 200,
            },
            # Live strategy endpoints
            {
                "endpoint": "/api/live/strategies",
                "method": "GET",
                "expected_status": 200,
            },
            # Live system endpoints
            {
                "endpoint": "/api/live/system/status",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/live/sources",
                "method": "GET",
                "expected_status": 200,
            },
            # AI Strategy endpoints
            {
                "endpoint": "/api/ai/strategies",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/ai/strategies/active",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/ai/strategies/performance",
                "method": "GET",
                "expected_status": 200,
            },
            # Trading endpoints
            {
                "endpoint": "/api/trading/orders",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/trading/positions",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/trading/balance",
                "method": "GET",
                "expected_status": 200,
            },
            # Market endpoints
            {
                "endpoint": "/api/market/prices",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/market/symbols",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/market/klines",
                "method": "GET",
                "expected_status": 200,
            },
            # Analytics endpoints
            {
                "endpoint": "/api/analytics/performance",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/analytics/portfolio",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/analytics/risk",
                "method": "GET",
                "expected_status": 200,
            },
            # Bot endpoints
            {"endpoint": "/api/bots", "method": "GET", "expected_status": 200},
            {
                "endpoint": "/api/bots/status",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/bots/performance",
                "method": "GET",
                "expected_status": 200,
            },
            # Signal endpoints
            {
                "endpoint": "/api/signals",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/signals/active",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/signals/history",
                "method": "GET",
                "expected_status": 200,
            },
            # Notification endpoints
            {
                "endpoint": "/api/notifications",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/notifications/settings",
                "method": "GET",
                "expected_status": 200,
            },
            # Auto trading endpoints
            {
                "endpoint": "/api/auto-trading/status",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/auto-trading/config",
                "method": "GET",
                "expected_status": 200,
            },
            {
                "endpoint": "/api/auto-trading/history",
                "method": "GET",
                "expected_status": 200,
            },
            # WebSocket endpoints (test connection)
            {"endpoint": "/ws/feed", "method": "GET", "expected_status": 200},
            # Root endpoint
            {"endpoint": "/", "method": "GET", "expected_status": 200},
        ]

    async def run_all_tests(self):
        """Run tests for all endpoints"""
        endpoints = self.get_all_endpoints()
        self.results["summary"]["total_endpoints"] = len(endpoints)

        logger.info(f"Starting health check for {len(endpoints)} endpoints...")

        # Test endpoints concurrently
        tasks = []
        for endpoint_info in endpoints:
            task = self.test_endpoint(
                endpoint_info["endpoint"],
                endpoint_info["method"],
                endpoint_info["expected_status"],
            )
            tasks.append(task)

        # Wait for all tests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            self.results["endpoints"].append(result)

        logger.info("Health check completed!")

    def generate_report(self) -> str:
        """Generate a comprehensive report"""
        summary = self.results["summary"]
        success_rate = (
            (summary["successful"] / summary["total_endpoints"]) * 100
            if summary["total_endpoints"] > 0
            else 0
        )

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ENDPOINT HEALTH CHECK REPORT                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 SUMMARY
═══════════
• Base URL: {self.base_url}
• Timestamp: {self.results['timestamp']}
• Total Endpoints Tested: {summary['total_endpoints']}
• Successful: {summary['successful']} ✅
• Failed: {summary['failed']} ❌
• Errors: {summary['errors']} ⚠️
• Success Rate: {success_rate:.1f}%

📋 DETAILED RESULTS
═══════════════════
"""

        # Group endpoints by status
        successful = [ep for ep in self.results["endpoints"] if ep["status"] == "SUCCESS"]
        failed = [ep for ep in self.results["endpoints"] if ep["status"] == "FAILED"]
        errors = [ep for ep in self.results["endpoints"] if ep["status"] == "ERROR"]

        if successful:
            report += "\n✅ SUCCESSFUL ENDPOINTS:\n"
            for ep in successful:
                report += f"  • {ep['endpoint']} ({ep['response_time']}s)\n"

        if failed:
            report += "\n❌ FAILED ENDPOINTS:\n"
            for ep in failed:
                report += f"  • {ep['endpoint']} - {ep.get('error', 'Unknown error')}\n"

        if errors:
            report += "\n⚠️ ERROR ENDPOINTS:\n"
            for ep in errors:
                report += f"  • {ep['endpoint']} - {ep.get('error', 'Unknown error')}\n"

        # Performance analysis
        if successful:
            response_times = [ep["response_time"] for ep in successful]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)

            report += f"""
📈 PERFORMANCE ANALYSIS
═══════════════════════
• Average Response Time: {avg_response_time:.3f}s
• Fastest Response: {min_response_time:.3f}s
• Slowest Response: {max_response_time:.3f}s
"""

        # Docker deployment recommendations
        report += """
🚀 DOCKER DEPLOYMENT RECOMMENDATIONS
═══════════════════════════════════════
"""

        if success_rate >= 95:
            report += "✅ READY FOR DEPLOYMENT - All critical endpoints are working\n"
        elif success_rate >= 80:
            report += "⚠️ DEPLOYMENT READY WITH WARNINGS - Some non-critical endpoints failed\n"
        else:
            report += "❌ NOT READY FOR DEPLOYMENT - Critical endpoints are failing\n"

        if failed or errors:
            report += "\n🔧 RECOMMENDED ACTIONS:\n"
            for ep in failed + errors:
                report += f"  • Fix {ep['endpoint']}: {ep.get('error', 'Unknown issue')}\n"

        report += """
╔══════════════════════════════════════════════════════════════════════════════╗
║                              END OF REPORT                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

        return report

    def save_report(self, filename: str = None):
        """Save the report to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"endpoint_health_report_{timestamp}.json"

        # Save JSON data
        json_path = Path(filename)
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save human-readable report
        txt_path = Path(filename.replace(".json", ".txt"))
        report = self.generate_report()
        with open(txt_path, "w") as f:
            f.write(report)

        logger.info(f"Reports saved to {json_path} and {txt_path}")
        return str(json_path), str(txt_path)


async def main():
    """Main function to run the health check"""
    import sys

    # Get base URL from command line argument or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    checker = EndpointHealthChecker(base_url)

    try:
        await checker.run_all_tests()
        report = checker.generate_report()
        print(report)

        # Save reports
        json_file, txt_file = checker.save_report()
        print("\n📁 Reports saved to:")
        print(f"  • JSON: {json_file}")
        print(f"  • Text: {txt_file}")

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
