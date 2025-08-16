import os, re, sys, ast

FUTURE_LINE = "from __future__ import annotations\n"

def _pos_to_index(text: str, row: int, col: int) -> int:
    lines = text.splitlines(True)
    return sum(len(lines[i]) for i in range(row-1)) + col

def find_insert_index(text: str) -> int:
    """Return index to insert future import (just after top-level docstring if present)."""
    try:
        mod = ast.parse(text)
    except SyntaxError:
        return 0
    if mod.body and isinstance(mod.body[0], ast.Expr):
        val = getattr(mod.body[0], "value", None)
        # Py3.10: Constant string for docstring
        if getattr(val, "value", None) is not None and isinstance(val.value, str):
            end_lineno = getattr(mod.body[0], "end_lineno", None)
            end_col = getattr(mod.body[0], "end_col_offset", None)
            if end_lineno is not None and end_col is not None:
                return _pos_to_index(text, end_lineno, end_col)
    return 0

def normalize_text(text: str) -> str:
    # Remove all occurrences anywhere to avoid duplicates
    text2 = re.sub(r'(?m)^[ \t]*from __future__ import annotations[ \t]*\r?\n', '', text)

    insert_at = find_insert_index(text2)

    head = text2[:insert_at]
    body = text2[insert_at:]

    # ensure neat spacing around the inserted line
    if head and not head.endswith("\n"):
        head += "\n"

    insert = FUTURE_LINE
    if body and not body.startswith("\n"):
        insert += "\n"

    return head + insert + body

def fix_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            orig = f.read()
    except Exception:
        return False

    new_text = normalize_text(orig)

    if new_text != orig:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_text)
        return True
    return False

def iter_python_files(root: str):
    # include app.py explicitly
    app_py = os.path.join(root, "app.py")
    if os.path.exists(app_py):
        yield app_py
    # and all pages/*
    pages_dir = os.path.join(root, "pages")
    for dp, _, fnames in os.walk(pages_dir):
        for name in fnames:
            if name.endswith(".py"):
                yield os.path.join(dp, name)

def main():
    roots = sys.argv[1:] or ["streamlit"]
    changed = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for path in iter_python_files(r):
            if fix_file(path):
                changed.append(path)
    print(f"Updated {len(changed)} file(s).")
    for p in changed:
        print("  -", p)

if __name__ == "__main__":
    main()
