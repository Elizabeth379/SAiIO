import networkx as nx
from methods import ford_fulkerson_algorithm

if __name__ == "__main__":
    G = nx.DiGraph()

    G.add_edge('s', 'a', capacity=3)
    G.add_edge('s', 'b', capacity=2)
    G.add_edge('a', 'b', capacity=2)
    G.add_edge('a', 't', capacity=1)
    G.add_edge('b', 't', capacity=2)

    result = ford_fulkerson_algorithm(G, 's', 't')
    print(result)