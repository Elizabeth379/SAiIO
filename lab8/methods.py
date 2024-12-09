from collections import deque
from copy import deepcopy
import networkx as nx


def label_method(Gf: nx.DiGraph, start: str):
    Q = deque([start])
    X = set([start])
    l = {start: None}

    while Q:
        u = Q.popleft()
        for v in Gf.neighbors(u):
            if v not in X and Gf[u][v]["capacity"] > 0:
                l[v] = (u, v)
                X.add(v)
                Q.append(v)

    return X, l


def ford_fulkerson_algorithm(G: nx.DiGraph, start: str, end: str) -> int:
    # Step 0
    edges = deepcopy(G.edges())
    for edge in edges:
        if (edge[1], edge[0]) not in edges:
            G.add_edge(edge[1], edge[0], capacity=0)
    max_flow = 0

    # Step 1
    f = {edge: 0 for edge in G.edges()}

    # Step 2
    Gf = deepcopy(G)
    for u, v in Gf.edges():
        Gf[u][v]["capacity"] = G[u][v]["capacity"] - f[(u, v)] + f[(v, u)]

    while True:
        # Step 3
        X, l = label_method(Gf, start)
        if end not in X:
            break

        # Step 4
        path, v = [], end
        while True:
            u, v = l[v]
            path.append((u, v))
            v = u
            if v == start:
                break
        path.reverse()

        # Step 5
        theta = min(G[u][v]["capacity"] for u, v in path)

        # Step 6
        fP = {edge: (theta if edge in path else 0) for edge in G.edges()}

        # Step 7 + 8
        for u, v in path:
            f[(u, v)] += fP[(u, v)]
            Gf[u][v]["capacity"] -= theta
            Gf[v][u]["capacity"] += theta
        max_flow += theta

    return max_flow, f