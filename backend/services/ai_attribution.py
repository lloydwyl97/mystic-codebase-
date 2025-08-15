import json
from datetime import datetime, timezone
from typing import Any, Dict

from redis import Redis

redis_client: Redis = Redis.from_url("redis://localhost:6379/0")


def save_attribution(symbol: str, inputs: Dict[str, Any], weights: Dict[str, float], reason: str) -> None:
    data: Dict[str, Any] = {
        "used": {"inputs": inputs, "weights": weights, "reason": reason},
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    redis_client.set(f"ai:attr:{symbol}", json.dumps(data), ex=600)


def load_attribution(symbol: str) -> Dict[str, Any]:
    raw = redis_client.get(f"ai:attr:{symbol}")
    if raw:
        try:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            return json.loads(text)
        except Exception:
            pass
    return {"used": {}, "ts": datetime.now(timezone.utc).isoformat()}


