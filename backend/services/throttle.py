import time

last_call: dict[str, float] = {}
limits: dict[str, float] = {}


def setup_limits(provider_limits: dict[str, int]) -> None:
    global limits
    limits = {k: 60 / v for k, v in provider_limits.items()}  # seconds between requests


def throttle(provider: str) -> None:
    now = time.time()
    wait = limits.get(provider, 2)
    last = last_call.get(provider, 0)
    diff = now - last
    if diff < wait:
        time.sleep(wait - diff)
    last_call[provider] = time.time()


