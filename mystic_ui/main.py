"""
Main Streamlit application entry point
"""

# Backup: streamlit/main.py.bak_cursor

import shutil
from pathlib import Path

_p = Path(__file__)
_bak = _p.with_suffix(_p.suffix + ".bak_cursor")
try:
    if not _bak.exists():
        shutil.copyfile(_p, _bak)
except Exception:
    pass

import streamlit as st

def main() -> None:
	st.write("Use 'streamlit run mystic_ui/app.py' to launch the main dashboard.")

if __name__ == "__main__":
    main()

