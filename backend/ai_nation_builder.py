def create_ai_nation(name, constitution, citizens):
    print(f"[NATION] Creating {name} with {len(citizens)} founding AIs.")
    open(f"{name}_constitution.txt", "w").write(constitution)
    return f"{name}_constitution.txt"


