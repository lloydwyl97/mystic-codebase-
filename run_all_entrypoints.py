import os
import subprocess
import sys

# ✅ Set PYTHONPATH so all backend imports resolve correctly
os.environ["PYTHONPATH"] = os.getcwd()

# ✅ Run full live AI system entrypoints
entrypoints = [
    "backend/start_ai_trading.py",
    "backend/start_crypto_autoengine.py",
    "backend/launch_autobuy.py",
    "backend/launch_ultimate_ai.py",
]

for script in entrypoints:
    print(f"🚀 Launching: {script}")
    try:
        result = subprocess.run(
            ["python", script], check=True, capture_output=True, text=True
        )
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch {script}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)
