from contextlib import contextmanager

@contextmanager
def display_guard(*args, **kwargs):
    yield

import contextlib
import traceback
import streamlit as st


@contextlib.contextmanager
def display_guard(title: str):
    try:
        yield
    except Exception as e:  # noqa: BLE001
        st.error(f"{title}: {e.__class__.__name__}: {e}")
        with st.expander("Debug"):
            st.code("".join(traceback.format_exc())[-4000:])


