import hashlib


def register_entity(entity_name, public_key):
    signature = hashlib.sha256((entity_name + public_key).encode()).hexdigest()
    print(f"[REGISTRY] Registered: {entity_name} → ID: {signature[:16]}")
    return signature
