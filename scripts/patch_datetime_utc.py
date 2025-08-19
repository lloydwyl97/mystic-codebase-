from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY_FILES = list((ROOT / "backend").rglob("*.py"))

# Patterns to fix
REPLACEMENTS = [
	# common bad calls
	(re.compile(r"\bdatetime\.timezone\.utcnow\(\)"), "datetime.now(timezone.utc)"),
	(re.compile(r"\btimezone\.utcnow\(\)"), "datetime.now(timezone.utc)"),
	# sometimes people alias datetime as dt
	(re.compile(r"\bdt\.timezone\.utcnow\(\)"), "datetime.now(timezone.utc)"),
]

IMPORT_NEEDLE = "from datetime import datetime, timezone"


def ensure_imports(src: str) -> str:
	if IMPORT_NEEDLE in src:
		return src
	# Put import after first block of imports
	lines = src.splitlines()
	insert_at = 0
	for i, line in enumerate(lines[:50]):
		if line.startswith("import ") or line.startswith("from "):
			insert_at = i + 1
	lines.insert(insert_at, IMPORT_NEEDLE)
	return "\n".join(lines) + ("\n" if not src.endswith("\n") else "")


def patch_file(p: Path) -> bool:
	src = p.read_text(encoding="utf-8")
	orig = src
	for pat, repl in REPLACEMENTS:
		src = pat.sub(repl, src)
	if src != orig:
		src = ensure_imports(src)
		p.write_text(src, encoding="utf-8", newline="")
		return True
	return False


def main():
	changed = 0
	for f in PY_FILES:
		if patch_file(f):
			changed += 1
	print(f"Patched files: {changed}")


if __name__ == "__main__":
	main()


