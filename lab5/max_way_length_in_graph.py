import numpy as np
from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(dict)
        self.set_of_vertices = set()
        self.number_of_vertices = 0
        self.topological_sort_list = list()

    def add_edge(self, u, v, weight):
        if v in self.graph[u]:
            weight = max(weight, self.graph[u][v])
        self.graph[u].update({v: weight})
        self.set_of_vertices.add(u)
        self.set_of_vertices.add(v)
        self.number_of_vertices = len(self.set_of_vertices)

    def topological_sort_util(self, v, visited, stack):
        visited[v] = True

        for i in self.graph[v].keys():
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)

        stack.insert(0, v)

    def topological_sort(self):
        visited = [False for _ in range(self.number_of_vertices)]
        stack = list()

        for i in range(self.number_of_vertices):
            if not visited[i]:
                self.topological_sort_util(i, visited, stack)

        self.topological_sort_list = stack.copy()

    def get_max_path_len(self, start, stop):
        opt = dict(zip(self.topological_sort_list, [-np.inf for _ in range(self.number_of_vertices)]))
        opt[start] = 0
        x = dict(zip(self.topological_sort_list, [None for _ in range(self.number_of_vertices)]))

        for i in range(1, self.number_of_vertices):
            cur_peak = self.topological_sort_list[i]
            prev_peak = self.topological_sort_list[i-1]
            max_path_to_current_peak = -np.inf
            for j in range(i):
                peak_before_cur = self.topological_sort_list[j]
                if cur_peak in self.graph[peak_before_cur]:
                    cur_path = opt[peak_before_cur] + self.graph[peak_before_cur][cur_peak]
                    if max_path_to_current_peak < cur_path:
                        max_path_to_current_peak = cur_path
                        prev_peak = peak_before_cur

            opt[cur_peak] = max_path_to_current_peak
            x[cur_peak] = prev_peak

        revers_restored_path = list([stop])
        while x[revers_restored_path[-1]] is not None:
            revers_restored_path.append(x[revers_restored_path[-1]])

        return revers_restored_path[::-1], opt[stop]