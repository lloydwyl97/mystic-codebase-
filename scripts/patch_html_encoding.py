from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = []
FILES += list((ROOT / "backend").rglob("*.html"))
FILES += list((ROOT / "backend").rglob("*.py"))  # also fix inline HTML in .py

# Insert <meta charset="utf-8"> right after <head> if missing
HEAD_OPEN = re.compile(r"(<head[^>]*>)", re.IGNORECASE)
HAS_META = re.compile(r"<meta\s+charset=['\"]?utf-?8['\"]?\s*/?>", re.IGNORECASE)

# Common mojibake map (extend as needed)
MOJIBAKE = {
	"â€¦": "…",
	"â€”": "—",
	"â€“": "–",
	"â€˜": "‘",
	"â€™": "’",
	"â€œ": "“",
	"â€�": "”",
	"â€¢": "•",
	"â€": "†",
}


def fix_meta(text: str) -> str:
	if "<head" in text.lower() and not HAS_META.search(text):
		text = HEAD_OPEN.sub(r"\1\n    <meta charset=\"utf-8\">", text, count=1)
	return text


def fix_mojibake(text: str) -> str:
	for bad, good in MOJIBAKE.items():
		if bad in text:
			text = text.replace(bad, good)
	return text


def is_htmly(text: str) -> bool:
	# rough heuristic: contains html tags
	low = text.lower()
	return "<html" in low or "<head" in low or "<body" in low


def main():
	changed = 0
	for f in FILES:
		txt = f.read_text(encoding="utf-8", errors="ignore")
		orig = txt
		# only enforce meta on files that look like html (templates or inline html)
		if is_htmly(txt):
			txt = fix_meta(txt)
		# fix common mojibake everywhere
		txt = fix_mojibake(txt)
		if txt != orig:
			f.write_text(txt, encoding="utf-8", newline="")
			changed += 1
	print(f"Patched files: {changed}")


if __name__ == "__main__":
	main()


