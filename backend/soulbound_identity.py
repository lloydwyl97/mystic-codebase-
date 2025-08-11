import hashlib
import uuid


def generate_soul_signature(seed=None):
    unique_id = seed or str(uuid.uuid4())
    hash_id = hashlib.sha256(unique_id.encode()).hexdigest()
    return f"SOULBOUND-{hash_id[:16]}"
