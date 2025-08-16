import os
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

def _read_env_value(env_path: Path, key: str) -> Optional[str]:
    """Read a single KEY=value from .env if present (no third-party deps)."""
    try:
        if not env_path.exists():
            return None
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip()
    except Exception:
        pass
    return None

class Settings:
    """Small, explicit settings object. No pydantic to avoid version issues."""
    def __init__(self) -> None:
        # Root .env (project root = one dir above /backend)
        root = Path(__file__).resolve().parents[1]
        env_path = root / ".env"

        # Binance US
        self.binance_us_api_key = (
            os.getenv("BINANCE_US_API_KEY")
            or _read_env_value(env_path, "BINANCE_US_API_KEY")
            or ""
        )
        self.binance_us_api_secret = (
            os.getenv("BINANCE_US_API_SECRET")
            or os.getenv("BINANCE_US_SECRET_KEY")  # fallback key name
            or _read_env_value(env_path, "BINANCE_US_API_SECRET")
            or _read_env_value(env_path, "BINANCE_US_SECRET_KEY")
            or ""
        )

        # Coinbase (kept for completeness)
        self.coinbase_api_key = os.getenv("COINBASE_API_KEY") or ""
        self.coinbase_api_secret = (
            os.getenv("COINBASE_API_SECRET")
            or os.getenv("COINBASE_SECRET_KEY")
            or ""
        )

        # Display defaults
        self.display_exchange = os.getenv("DISPLAY_EXCHANGE", "binance.us")
        try:
            self.display_top_n = int(os.getenv("DISPLAY_TOP_N", "10"))
        except Exception:
            self.display_top_n = 10

        # Back-compat wrapper so legacy code can do settings.exchange.<...>
        self.exchange = SimpleNamespace(
            name=self.display_exchange,
            binance_us_api_key=self.binance_us_api_key,
            binance_us_api_secret=self.binance_us_api_secret,
            # Legacy alias expected by some parts of the codebase
            binance_us_secret_key=self.binance_us_api_secret,
            coinbase_api_key=self.coinbase_api_key,
            coinbase_api_secret=self.coinbase_api_secret,
            # Optional legacy alias for Coinbase
            coinbase_secret_key=self.coinbase_api_secret,
        )
        self.exchange_json = {"name": self.exchange.name}

settings = Settings()

