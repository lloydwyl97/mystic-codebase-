def validate_trade(agent_a, agent_b, terms):
    print(f"[LAW] Reviewing trade between {agent_a} and {agent_b}")
    if terms["fairness"] > 0.8:
        print("[LAW] Approved ✔️")
        return True
    print("[LAW] Denied ❌")
    return False
