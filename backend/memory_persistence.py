import gzip
import pickle


def save_agent_memory(agent_id, memory):
    with gzip.open(f"{agent_id}_memory.pgz", "wb") as f:
        pickle.dump(memory, f)
    print(f"[MEMORY] {agent_id} saved to disk.")


def load_agent_memory(agent_id):
    with gzip.open(f"{agent_id}_memory.pgz", "rb") as f:
        return pickle.load(f)


