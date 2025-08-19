from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_FILES = list((ROOT / "backend").rglob("*.py"))

# Matches APIRouter(... prefix='/api' ...) with single or double quotes, any kwargs order
ROUTER_RE = re.compile(r"APIRouter\((?P<inside>[^)]*)\)", re.DOTALL)
PREFIX_RE = re.compile(r"prefix\s*=\s*(['\"])\/api\1")


def patch_router_calls(src: str) -> str:
    def repl(match: re.Match) -> str:
        inside = match.group("inside")
        if PREFIX_RE.search(inside):
            inside_new = PREFIX_RE.sub("prefix=\"\"", inside)
            return f"APIRouter({inside_new})"
        return match.group(0)

    return ROUTER_RE.sub(repl, src)


def patch_file(p: Path) -> bool:
    txt = p.read_text(encoding="utf-8", errors="ignore")
    new_txt = patch_router_calls(txt)
    if new_txt != txt:
        p.write_text(new_txt, encoding="utf-8", newline="")
        return True
    return False


def main() -> None:
    changed = 0
    for f in PY_FILES:
        if patch_file(f):
            changed += 1
    print(f"Patched router files: {changed}")


if __name__ == "__main__":
    main()


