import graph

import random


class SpanningTree(graph.UndirectedGraph):
    # def __init__(self, g: graph.Graph):

    def build_spanning_tree_from_graph(self, g: graph.Graph):
        """
        Builds a spanning tree starting by an undirected graph. Implemented through Prim's algorithm.
        """

        if len(g.get_nodes()) == 0 or not g.is_graph_connected() or not type(g) == graph.UndirectedGraph:
            return None

        graph_nodes: list[int] = g.get_nodes()
        starting_node: int = g.get_nodes()[random.randint(0, len(graph_nodes) - 1)]
        
        nodes: list[int] = [starting_node]
        edges: dict[int, list[tuple[int, int]]] = {}

        while len(nodes) < len(graph_nodes):
            outcoming_cut = g.get_outcoming_cut(nodes)

            dict_of_minimum = {key: min(vals, key=lambda x: x[1]) for key, vals in outcoming_cut.items()}
            minimum_cost_node = min(dict_of_minimum, key=lambda x: dict_of_minimum[x][1])
            minimum_cost_edge = dict_of_minimum[minimum_cost_node]

            nodes.append(minimum_cost_edge[0])

            if minimum_cost_node in edges:
                edges[minimum_cost_node].append(minimum_cost_edge)
            else:
                edges[minimum_cost_node] = [minimum_cost_edge]

            if minimum_cost_edge[0] in edges:
                edges[minimum_cost_edge[0]].append((minimum_cost_node, minimum_cost_edge[1]))
            else:
                edges[minimum_cost_edge[0]] = [(minimum_cost_node, minimum_cost_edge[1])]

        return edges


g = graph.DirectedGraph()
g.add_edge(1,2)
g.add_edge(2,1)

st = SpanningTree()
result = st.build_spanning_tree_from_graph(g)

print(f"original tree: {g.edges}")
print(f"spanning tree: {result}")