import networkx as nx
from itertools import chain, combinations
import numpy as np

def jaccard_set(set1, set2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(set1.intersection(set2))
    union = (len(set1) + len(set2)) - intersection
    return float(intersection) / union

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


def generate_jaccard(L):
    #wegihted graph with intersection over union as weight
    """Generate weighted diagram"""

    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    u = list(powerset(L))
    n = len(u)
    forward_dict = dict(zip(range(n), u))
    backward_dict = dict(zip(u, range(n)))
    graph = nx.Graph()
    graph.add_nodes_from(u)
    list_w = []
    for i in range(n):
        for j in range(i+1, n):
            w = jaccard_set(set(u[i]), set(u[j]))
            list_w.append(w)
    median = np.median(np.array(list_w))
    
    for i in range(n):
        for j in range(i+1, n):
            w = jaccard_set(set(u[i]), set(u[j]))
            if w > median:
                graph.add_edge(u[i], u[j])
    graph = nx.relabel_nodes(graph, backward_dict)
    return graph, forward_dict, backward_dict

if __name__ == "__main__":
    L = range(4)
    generate_jaccard(L)