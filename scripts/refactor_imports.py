import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# map old import -> new import
RENAMES = {
    # persistent_cache
    r'\bfrom\s+backend\.ai\.persistent_cache\s+import\b': 'from backend.modules.ai.persistent_cache import',
    r'\bfrom\s+ai\.persistent_cache\s+import\b': 'from backend.modules.ai.persistent_cache import',
    r'\bfrom\s+modules\.ai\.persistent_cache\s+import\b': 'from backend.modules.ai.persistent_cache import',
    r'\bfrom\s+ai\.ai\.persistent_cache\s+import\b': 'from backend.modules.ai.persistent_cache import',

    r'\bimport\s+backend\.ai\.persistent_cache\b': 'import backend.modules.ai.persistent_cache',
    r'\bimport\s+ai\.persistent_cache\b': 'import backend.modules.ai.persistent_cache',
    r'\bimport\s+modules\.ai\.persistent_cache\b': 'import backend.modules.ai.persistent_cache',
    r'\bimport\s+ai\.ai\.persistent_cache\b': 'import backend.modules.ai.persistent_cache',

    # ai_brains
    r'\bfrom\s+backend\.modules\.ai\.ai_brains\s+import\b': 'from backend.modules.ai.ai_brains import',
    r'\bfrom\s+ai\.ai\.ai_brains\s+import\b':               'from backend.modules.ai.ai_brains import',

    r'\bimport\s+ai\.ai\.ai_brains\b':                      'import backend.modules.ai.ai_brains',

    # ai_breakouts
    r'\bfrom\s+backend\.modules\.ai\.ai_breakouts\s+import\b': 'from backend.modules.ai.ai_breakouts import',
    r'\bfrom\s+ai\.ai\.ai_breakouts\s+import\b':               'from backend.modules.ai.ai_breakouts import',

    # analytics_engine
    r'\bfrom\s+backend\.modules\.metrics\.analytics_engine\s+import\b': 'from backend.modules.ai.analytics_engine import',
    r'\bfrom\s+modules\.ai\.analytics_engine\s+import\b':               'from backend.modules.ai.analytics_engine import',
    r'\bfrom\s+backend\.modules\.ai\.analytics_engine\s+import\b':      'from backend.modules.ai.analytics_engine import',

    # coins
    r'\bfrom\s+config\.coins\s+import\b':                  'from backend.config.coins import',
}


def rewrite_file(path: pathlib.Path):
    txt = path.read_text(encoding='utf-8', errors='ignore')
    orig = txt
    for pattern, repl in RENAMES.items():
        txt = re.sub(pattern, repl, txt)
    if txt != orig:
        path.write_text(txt, encoding='utf-8')
        print(f"REWROTE {path}")


def main():
    for dirpath, _, filenames in os.walk(ROOT):
        # skip vendored/build dirs
        if any(skip in dirpath for skip in ('node_modules', 'frontend/dist', '.venv', 'venv', '__pycache__')):
            continue
        for fn in filenames:
            if fn.endswith('.py'):
                rewrite_file(pathlib.Path(dirpath) / fn)


if __name__ == "__main__":
    main()


