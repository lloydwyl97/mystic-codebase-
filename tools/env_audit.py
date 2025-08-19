# tools/env_audit.py
import os

from dotenv import dotenv_values

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_DIRS = ["backend", "ai", "middleware", "alerts", "frontend"]


def audit_env_files():
    report = []
    for module in MODULE_DIRS:
        env_path = os.path.join(ROOT_DIR, module, ".env")
        if not os.path.exists(env_path):
            report.append(f"❌ MISSING: {module}/.env")
            continue

        config = dotenv_values(env_path)
        if not config:
            report.append(f"⚠️ EMPTY: {module}/.env")
        else:
            report.append(f"✅ {module}/.env — {len(config)} keys")
            for k, v in config.items():
                report.append(f"   • {k} = {v or '[NO VALUE]'}")

    return "\n".join(report)


if __name__ == "__main__":
    print("🔍 ENV FILE AUDIT REPORT:\n")
    print(audit_env_files())
