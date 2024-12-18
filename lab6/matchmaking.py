import pprint
import matplotlib.pyplot as plt
import numpy as np


def bfs(graph, s, t, parent):
    visited = [False] * len(graph)
    queue = []

    queue.append(s)
    visited[s] = True

    while queue:
        u = queue.pop(0)

        for ind, val in enumerate(graph[u]):
            if visited[ind] == False and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u

    return True if visited[t] else False


def matchmaking(graph, source, sink):
    parent = [-1] * len(graph)
    max_flow = 0
    steps = []

    while bfs(graph, source, sink, parent):
        path_flow = float('Inf')
        s = sink
        while (s != source):
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]

        max_flow += path_flow

        v = sink
        while (v != source):
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
        steps.append(parent)

    return max_flow, graph, steps