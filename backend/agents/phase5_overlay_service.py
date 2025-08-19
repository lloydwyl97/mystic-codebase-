import asyncio

import redis.asyncio as redis

r = redis.Redis(decode_responses=True)


async def fetch_metric(key, default="..."):
    try:
        value = await r.get(key)
        return value or default
    except Exception:
        return default


def get_phase5_overlay_metrics():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    neuro_sync = loop.run_until_complete(fetch_metric("neuro_sync_index"))
    cosmic_signal = loop.run_until_complete(fetch_metric("cosmic_harmonic_status"))
    aura_alignment = loop.run_until_complete(fetch_metric("aura_alignment_score"))
    interdim_activity = loop.run_until_complete(fetch_metric("interdim_signal_strength"))

    return {
        "neuro_sync": neuro_sync,
        "cosmic_signal": cosmic_signal,
        "aura_alignment": aura_alignment,
        "interdim_activity": interdim_activity,
    }


