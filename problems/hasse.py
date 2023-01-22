import networkx as nx
from itertools import chain, combinations

# chain and combinations is from itertools
def generate_hasse(L):
    """Generate Hasse diagram"""

    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    u = list(powerset(L))
    
    forward_dict = dict(zip(range(len(u)), u))
    backward_dict = dict(zip(u, range(len(u))))

    graph = nx.DiGraph()
    for node_tiny in u:
        for node_big in u:
            if len(node_tiny) == len(node_big)-1:
                if frozenset(node_tiny) < frozenset(node_big):
                    graph.add_edge(node_tiny, node_big)

    graph = nx.relabel_nodes(graph, backward_dict)
    return graph, forward_dict, backward_dict