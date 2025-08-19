import json
import os
import re
import sys
from glob import glob
from importlib.metadata import distributions


def list_requirement_files(root: str) -> list[str]:
    files: set[str] = set(glob(os.path.join(root, "**", "*requirements*.txt"), recursive=True))
    req_txt = os.path.join(root, "requirements.txt")
    if os.path.exists(req_txt):
        files.add(req_txt)
    return sorted(files)


def parse_requirements(file_path: str) -> list[str]:
    packages: set[str] = set()
    name_re = re.compile(r"^[A-Za-z0-9_.-]+")
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("") or line.startswith("--"):
                    continue
                if line.startswith("git+") or line.startswith("http"):
                    # URL VCS installs; skip name extraction
                    continue
                token = line.split(";", 1)[0].strip()
                token = token.split(" ", 1)[0].strip()
                token = token.split("[", 1)[0].strip()
                token = re.split(r"(==|>=|<=|~=|!=|===|>|<)", token)[0].strip()
                m = name_re.match(token)
                if m:
                    packages.add(m.group(0).lower())
    except Exception:
        # unreadable or malformed file → treat as empty
        return []
    return sorted(packages)


def list_installed() -> set[str]:
    out: set[str] = set()
    for d in distributions():
        try:
            name = (d.metadata.get("Name") or "").strip()
            if name:
                out.add(name.lower())
        except Exception:
            continue
    return out


def main() -> None:
    root = os.getcwd()
    files = list_requirement_files(root)
    installed = list_installed()

    report: dict[str, object] = {
        "installed_count": len(installed),
        "files": {},
        "summary": {"total_files": 0, "unique_required": 0, "unique_missing": 0},
    }

    all_required: set[str] = set()
    all_missing: set[str] = set()

    for f in files:
        pkgs = parse_requirements(f)
        missing = sorted([p for p in pkgs if p not in installed])
        present_count = sum(1 for p in pkgs if p in installed)
        report["files"][os.path.relpath(f, root)] = {
            "count": len(pkgs),
            "missing_count": len(missing),
            "missing": missing,
            "present_count": present_count,
        }
        all_required.update(pkgs)
        all_missing.update(missing)

    report["summary"]["total_files"] = len(files)
    report["summary"]["unique_required"] = len(all_required)
    report["summary"]["unique_missing"] = len(all_missing)

    json.dump(report, sys.stdout, indent=2)


if __name__ == "__main__":
    main()


