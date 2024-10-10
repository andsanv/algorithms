from data_structures.graphs import graph



def compute_shortest_path(graph: graph.Graph, starting_node: int):
    if any(map(lambda x: x[1] < 0, graph.edges.values())):  # Dijkstra cannot be applied if graph contains negative edges
        return None
    
    if starting_node not in graph.get_nodes():  
        return None
    
    costs: dict[int, tuple[int, int]] = {}  # dict containing, for each node, its precedessor and the cost

    for node in graph:  # setting initial cost to infinite
        if node != starting_node:
            costs[node] = (None, float('inf'))

    print(costs)