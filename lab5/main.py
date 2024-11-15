from max_way_length_in_graph import Graph


def main() -> None:
    s: int = 0
    t: int = 4
    graph: Graph = Graph()

    graph.add_edge(0, 1, 5)
    graph.add_edge(0, 3, 3)
    graph.add_edge(1, 2, 3)
    graph.add_edge(1, 3, 1)
    graph.add_edge(3, 4, 4)
    graph.add_edge(4, 2, 2)

    graph.topological_sort()

    if s not in graph.topological_sort_list or t not in graph.topological_sort_list or graph.topological_sort_list.index(s) > graph.topological_sort_list.index(t):
        print('No path from s to t')
    else:
        print(f'Topological sort: {graph.topological_sort_list}')

        max_path_from_start_to_stop, path_length = graph.get_max_path_len(s, t)
        #print(f'max ({s}, {t}) path: {max_path_from_start_to_stop[0]}', end='')
        #for i in range(1, len(max_path_from_start_to_stop)):
        #    print(f' -> {max_path_from_start_to_stop[i]}', end='')

        print(f'\nPath length: {path_length}')


if __name__ == '__main__':
    main()
