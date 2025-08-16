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

# Inject theme before importing heavy pages
from streamlit.ui.theme import inject_global_theme
inject_global_theme()

# Import entrypoint (fallback to no-op if module missing)
try:
	from streamlit._pages.mystic_super_dashboard import main  # type: ignore[attr-defined]
except Exception:
	def main():  # type: ignore[no-redef]
		return None

if __name__ == "__main__":
    main()
