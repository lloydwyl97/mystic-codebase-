import shutil
import os


def replicate_to(path="./replica", dna="core"):
    """Replicate the system to a new location"""
    os.makedirs(path, exist_ok=True)

    # Get current directory
    current_dir = os.getcwd()

    # Copy current directory to replica
    try:
        shutil.copytree(current_dir, f"{path}/{dna}", dirs_exist_ok=True)
        print(f"[REPLICATOR] {dna} replicated to {path}")
    except Exception as e:
        print(f"[REPLICATOR] Replication failed: {e}")
        # Create a simple backup instead
        with open(f"{path}/{dna}_backup.txt", "w") as f:
            f.write(f"System backup created at {path}/{dna}")
        print("[REPLICATOR] Created backup file instead")


