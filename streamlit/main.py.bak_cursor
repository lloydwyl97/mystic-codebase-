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

from streamlit.pages.mystic_super_dashboard import main

if __name__ == "__main__":
    main()
