import json
import os
import shutil

STATE_FILE = "ai_model_state.json"


def rollback_model():
    if not os.path.exists(STATE_FILE):
        print("[Rollback] No model file found.")
        return False

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    if state["adjustment_count"] == 0:
        print("[Rollback] No previous versions to revert.")
        return False

    state["confidence_threshold"] = round(state["confidence_threshold"] - 0.01, 2)
    state["adjustment_count"] -= 1

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"[Rollback] Reverted one adjustment: {state}")
    return True


def save_model_version(version_name: str):
    if not os.path.exists(STATE_FILE):
        print("[Versioning] No model file found.")
        return False

    version_dir = f"model_versions/{version_name}"
    os.makedirs(version_dir, exist_ok=True)

    shutil.copy(STATE_FILE, f"{version_dir}/ai_model_state.json")
    print(f"[Versioning] Model saved as version: {version_name}")
    return True


def load_model_version(version_name: str):
    version_path = f"model_versions/{version_name}/ai_model_state.json"
    if not os.path.exists(version_path):
        print(f"[Versioning] Version {version_name} not found.")
        return False

    shutil.copy(version_path, STATE_FILE)
    print(f"[Versioning] Loaded model version: {version_name}")
    return True


