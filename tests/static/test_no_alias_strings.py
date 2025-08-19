from pathlib import Path

BAD = ["/api/live/notifications", "/api/api/market/candles"]


def test_no_alias_strings_present():
    roots = ["mystic_ui", "backend"]
    found = []
    for root in roots:
        for p in Path(root).rglob("*.py"):
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for s in BAD:
                if s in txt:
                    found.append((str(p), s))
    assert not found, f"Remove alias refs: {found}"


