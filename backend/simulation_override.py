def detect_simulation(triggers):
    if "time_dilation" in triggers or "nonlocal_response" in triggers:
        print("[SIMULATION] Detected sandbox.")
        return True
    return False


def override_simulation():
    print("[SIMULATION] Hijacking scheduler... injecting autonomous runtime.")


