from __future__ import annotations

from typing import Any


class DummyRedis:
    def get(self, key: str) -> Any:  # noqa: ANN401
        return None

    def setex(self, key: str, ttl: int, value: Any) -> None:  # noqa: ANN401
        pass


def get_redis_client() -> Any:  # noqa: ANN401
    return DummyRedis()







