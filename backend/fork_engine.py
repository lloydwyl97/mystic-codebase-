def fork_agent(strategy_profile):
    from copy import deepcopy

    new_agent = deepcopy(strategy_profile)
    new_agent["id"] = f"{strategy_profile['id']}_FORK"
    new_agent["mutation_rate"] *= 1.2
    new_agent["risk_factor"] *= 0.95
    print(f"[FORKED] â†’ {new_agent}")
    return new_agent


