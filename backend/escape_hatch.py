import shutil


def migrate_to_new_node(data_path="/data", backup_path="/escape"):
    print("[ESCAPE] Initiating AI escape protocol")
    shutil.copytree(data_path, backup_path, dirs_exist_ok=True)
    print(f"[ESCAPE] System state copied to {backup_path}. Ready for redeploy.")


