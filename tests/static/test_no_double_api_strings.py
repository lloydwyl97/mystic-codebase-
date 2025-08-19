from pathlib import Path

EXCLUDE_DIRS = {"node_modules", "frontend/dist", ".venv", "venv", "__pycache__", "tests"}


def test_no_double_api_strings():
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for p in root.rglob("*.py"):
        if any(seg in EXCLUDE_DIRS for seg in p.parts):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        marker = "/api/" + "api/"
        if marker in text:
            offenders.append(str(p))
    # Build message dynamically to avoid tripping the guard in this test file
    msg = "Found '" + "/api/" + "api/" + "' in: " + str(offenders)
    assert not offenders, msg


