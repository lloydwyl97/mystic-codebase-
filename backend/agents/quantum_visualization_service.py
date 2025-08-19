import asyncio

import numpy as np
import plotly.graph_objs as go
import redis.asyncio as redis
import streamlit as st

r = redis.Redis(decode_responses=True)


async def fetch(key, default="N/A"):
    try:
        val = await r.get(key)
        return val if val else default
    except Exception:
        return default


def render_quantum_indicators():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    q_signal = loop.run_until_complete(fetch("quantum_signal_level"))
    q_prob = loop.run_until_complete(fetch("quantum_trade_probability"))
    q_entropy = loop.run_until_complete(fetch("quantum_entropy_index"))

    st.markdown(
        f"""
        <div style='background-color:#1a1d25;padding:0.75em;border-radius:8px;margin-bottom:0.75em'>
            <span style='color:#7CFC00;font-weight:bold'>âš› Quantum Signal:</span> {q_signal} |
            <span style='color:#87CEFA;font-weight:bold'>ðŸ“ˆ Trade Probability:</span> {q_prob} |
            <span style='color:#FF8C00;font-weight:bold'>ðŸŒ€ Entropy Index:</span> {q_entropy}
        </div>
        """,
        unsafe_allow_html=True,
    )


async def get_waveform_data():
    try:
        raw = await r.lrange("quantum_waveform_data", 0, -1)
        waveform_data = []
        for x in raw[-300:]:
            try:
                value = float(x)
                # Validate for NaN and infinite values
                if not (np.isnan(value) or np.isinf(value)):
                    waveform_data.append(value)
                else:
                    # Replace invalid values with safe fallback
                    waveform_data.append(0.0)
            except (ValueError, TypeError):
                # Replace invalid data with safe fallback
                waveform_data.append(0.0)
        return waveform_data
    except Exception as e:
        print(f"Error fetching waveform data: {e}")
        return []


def render_quantum_waveform_chart():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    y = loop.run_until_complete(get_waveform_data())
    if not y:
        st.info("âš› No waveform data available yet.")
        return

    # Validate waveform data
    if any(np.isnan(y)) or any(np.isinf(y)):
        st.warning("âš ï¸ Invalid waveform data detected, using fallback")
        y = [0.0] * len(y) if y else [0.0] * 100

    x = list(range(len(y)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Quantum Signal"))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#FFFFFF",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


