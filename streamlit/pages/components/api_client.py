"""
🔌 API CLIENT - Centralized API communication for the dashboard
Handles all backend API calls with error handling and fallback data
"""

import requests
import streamlit as st
from typing import Dict, Any, Optional
import os


class APIClient:
    """Centralized API client for dashboard backend communication"""

    def __init__(self):
        # Get backend URL from environment or use default
        self.base_url = os.getenv("BACKEND_URL", "http://localhost:9000")
        self.timeout = 10  # 10 second timeout
        self.session = requests.Session()

        # Configure session headers
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "Mystic-Dashboard/2.0",
            }
        )

        # API path prefix - backend is mounted under a single /api prefix
        self.api_prefix = "/api"

        # Some calls may include double prefix; support gracefully
        self.triple_api_prefix = self.api_prefix + self.api_prefix

    def test_backend_connectivity(self) -> str:
        """Test backend connectivity and return status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return "Connected"
            else:
                return f"Error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Disconnected: {str(e)}"

    def fetch_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from API endpoint with error handling"""
        try:
            # Normalize endpoint to a single /api prefix
            if endpoint.startswith(self.triple_api_prefix + "/") or endpoint.startswith(self.api_prefix + "/"):
                url = f"{self.base_url}{endpoint}"
            else:
                if endpoint.startswith("/"):
                    url = f"{self.base_url}{self.api_prefix}{endpoint}"
                else:
                    url = f"{self.base_url}{self.api_prefix}/{endpoint}"

            response = self.session.get(url, timeout=self.timeout)

            # If non-200, try reasonable fallbacks for known route differences
            if response.status_code != 200:
                def build_url(path: str) -> str:
                    if path.startswith(self.triple_api_prefix + "/") or path.startswith(self.api_prefix + "/"):
                        return f"{self.base_url}{path}"
                    return f"{self.base_url}{self.api_prefix}{path if path.startswith('/') else '/' + path}"

                fallback_paths = []
                if endpoint.startswith("/trading/"):
                    fallback_paths.append("/live" + endpoint)
                if not endpoint.startswith("/api/"):
                    fallback_paths.append("/api" + endpoint)
                    # Some routers add an internal /api prefix resulting in /api/api/*
                    fallback_paths.append("/api/api" + endpoint)
                if endpoint == "/analytics/performance":
                    fallback_paths.append("/strategies/performance")
                    fallback_paths.append("/api/strategies/performance")
                    fallback_paths.append("/api/api/strategies/performance")
                if endpoint == "/risk/alerts":
                    fallback_paths.append("/risk/metrics")
                    fallback_paths.append("/api/risk/metrics")
                    fallback_paths.append("/api/api/risk/metrics")
                if endpoint == "/market/liquidity":
                    fallback_paths.append("/market/live")
                    fallback_paths.append("/api/market/live")

                for alt in fallback_paths:
                    try:
                        r2 = self.session.get(build_url(alt), timeout=self.timeout)
                        if r2.status_code == 200:
                            payload2 = r2.json()
                            return payload2 if (isinstance(payload2, dict) and "data" in payload2) else {"data": payload2}
                    except requests.exceptions.RequestException:
                        pass

            if response.status_code == 200:
                payload = response.json()
                return payload if (isinstance(payload, dict) and "data" in payload) else {"data": payload}
            else:
                st.warning(f"API Error: {response.status_code} for {endpoint}")
                return None

        except requests.exceptions.RequestException as e:
            st.warning(f"API Connection Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
            return None

    def post_api_data(
        self, endpoint: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Post data to API endpoint with error handling"""
        try:
            # Normalize endpoint to a single /api prefix
            if endpoint.startswith(self.triple_api_prefix + "/") or endpoint.startswith(self.api_prefix + "/"):
                url = f"{self.base_url}{endpoint}"
            else:
                if endpoint.startswith("/"):
                    url = f"{self.base_url}{self.api_prefix}{endpoint}"
                else:
                    url = f"{self.base_url}{self.api_prefix}/{endpoint}"

            response = self.session.post(url, json=data, timeout=self.timeout)

            if response.status_code in [200, 201]:
                payload = response.json()
                return payload if (isinstance(payload, dict) and "data" in payload) else {"data": payload}
            else:
                st.warning(f"API Error: {response.status_code} for {endpoint}")
                return None

        except requests.exceptions.RequestException as e:
            st.warning(f"API Connection Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
            return None

    def put_api_data(
        self, endpoint: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Put data to API endpoint with error handling"""
        try:
            # Normalize endpoint to a single /api prefix
            if endpoint.startswith(self.triple_api_prefix + "/") or endpoint.startswith(self.api_prefix + "/"):
                url = f"{self.base_url}{endpoint}"
            else:
                if endpoint.startswith("/"):
                    url = f"{self.base_url}{self.api_prefix}{endpoint}"
                else:
                    url = f"{self.base_url}{self.api_prefix}/{endpoint}"

            response = self.session.put(url, json=data, timeout=self.timeout)

            if response.status_code in [200, 201]:
                payload = response.json()
                return payload if (isinstance(payload, dict) and "data" in payload) else {"data": payload}
            else:
                st.warning(f"API Error: {response.status_code} for {endpoint}")
                return None

        except requests.exceptions.RequestException as e:
            st.warning(f"API Connection Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
            return None

    def delete_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Delete data from API endpoint with error handling"""
        try:
            # Normalize endpoint to a single /api prefix
            if endpoint.startswith(self.triple_api_prefix + "/") or endpoint.startswith(self.api_prefix + "/"):
                url = f"{self.base_url}{endpoint}"
            else:
                if endpoint.startswith("/"):
                    url = f"{self.base_url}{self.api_prefix}{endpoint}"
                else:
                    url = f"{self.base_url}{self.api_prefix}/{endpoint}"
            response = self.session.delete(url, timeout=self.timeout)

            if response.status_code in [200, 204]:
                if response.content:
                    try:
                        payload = response.json()
                        return payload if (isinstance(payload, dict) and "data" in payload) else {"data": payload}
                    except Exception:
                        return {}
                return {}
            else:
                st.warning(f"API Error: {response.status_code} for {endpoint}")
                return None

        except requests.exceptions.RequestException as e:
            st.warning(f"API Connection Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
            return None


# Global API client instance
api_client = APIClient()
