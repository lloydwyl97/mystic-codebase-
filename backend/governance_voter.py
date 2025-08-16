def evaluate_vote(proposal_text, impact_score, risk_score):
    decision = "YES" if impact_score > risk_score else "NO"
    print(f"[GOVERNANCE] Proposal: {proposal_text}")
    print(f"[GOVERNANCE] Vote: {decision}")
    return decision


