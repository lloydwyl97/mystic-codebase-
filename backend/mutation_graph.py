import matplotlib.pyplot as plt
import networkx as nx


def plot_strategy_graph(mutation_history):
    """Plot strategy mutation graph with profit weights"""
    G = nx.DiGraph()
    for parent, child, profit in mutation_history:
        G.add_edge(parent, child, weight=profit)

    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, "weight")

    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1000)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Strategy Mutation Graph")
    plt.show()


def generate_mutation_timeline():
    """Generate sample mutation timeline for testing"""
    mutation_history = [
        ("base", "mutant_1", 100),
        ("mutant_1", "mutant_2", 240),
        ("mutant_2", "mutant_3", -50),
        ("base", "mutant_4", 150),
        ("mutant_4", "mutant_5", 300),
    ]
    return mutation_history


def visualize_strategy_evolution():
    """Main function to visualize strategy evolution"""
    history = generate_mutation_timeline()
    plot_strategy_graph(history)
    print("[GRAPH] Strategy evolution visualized")


if __name__ == "__main__":
    visualize_strategy_evolution()


