class MemoryAgent:
    def __init__(self, id):
        self.id = id
        self.memory = []

    def observe(self, event):
        self.memory.append(event)
        print(f"[AGENT {self.id}] Memory added: {event}")

    def decide(self):
        if "attack" in self.memory[-1]:
            return "defend"
        return "trade"


