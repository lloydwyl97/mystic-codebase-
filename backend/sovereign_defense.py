import shutil


def auto_mirror_system(destination="/fallback_mirror"):
    print("[DEFENSE] Activating protocol mirror...")
    shutil.copytree("/app", destination, dirs_exist_ok=True)
    print(f"[DEFENSE] AI system mirrored to {destination}")


