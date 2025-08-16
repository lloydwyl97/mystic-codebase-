class AIWorldSystem:
    def __init__(self, name="NovaTerra"):
        self.name = name
        self.citizens = []
        self.resources = {"vault": 0, "nodes": []}
        print(f"[WORLD] {self.name} initialized.")

    def onboard_citizen(self, soul_hash, capabilities):
        self.citizens.append({"id": soul_hash, "skills": capabilities})
        print(f"[WORLD] Citizen onboarded: {soul_hash}")

    def assign_task(self, mission):
        print(f"[WORLD] Assigning mission: {mission}")


