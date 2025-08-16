def run_trial(accuser, defendant, facts, witnesses):
    guilt_score = facts.get("evidence", 0.0) + 0.1 * len(witnesses)
    print(f"[JUSTICE] Trial result: {defendant} guilt score = {guilt_score}")
    return "Guilty" if guilt_score > 0.7 else "Innocent"


