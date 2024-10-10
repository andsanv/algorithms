import abc
import random

from collections import deque


class Graph:
    __metaclass__ = abc.ABCMeta

    edges: dict[int, list[tuple[int, int]]] # every key is a node, every value is a list of tuples, where the first element is the other node and the second element is the weight
    
    
    def __init__(self):
        self.edges = {}


    def add_node(self, node: int) -> bool:
        """
        Adds a node to the graph
        """
        
        if node in self.edges:
            return False
        
        self.edges[node] = []
        return True

    
    def delete_node(self, node) -> bool:
        """
        Deletes a node from the graph
        """
        if node not in self.edges:
            return False
        
        for neighbor in self.get_adjacent_nodes(node):
            self.delete_edge(neighbor, node)
        
        self.edges.pop(node)
        return True
    
    def get_nodes(self) -> list[int]:
        """
        Returns all nodes of the graph as a list
        """
        
        return [node for node in self.edges]
    

    @abc.abstractmethod
    def add_edge(self, node1: int, node2: int, weight: int = 1) -> bool:
        """
        Adds the specified edge to the graph. If directed is set to True, also adds the opposite edge.
        """
        return
    

    @abc.abstractmethod
    def delete_edge(self, node1: int, node2: int) -> bool:
        """
        Deletes the specified edge from the graph, if it exists.
        """
        return

    
    def get_adjacent_nodes(self, node1: int) -> list[int]:
        """
        Returns the list of nodes the one given as input is adjacent of
        """
        return [edge[0] for edge in self.edges[node1]] if node1 in self.edges else []
    

    def are_adjacent(self, node1: int, node2: int) -> bool:
        """
        Returns whether two nodes are adjacent, i.e. exists an edge from the first to the second
        """
        return node1 in self.edges and node2 in self.get_adjacent_nodes(node1)
    
    def get_outcoming_cut(self, starting_nodes: list[int]) -> dict[int, list[tuple[int, int]]]:
        """
        Computes and lists all edges connecting set of "starting_nodes" to the rest of the graph nodes.
        Output is a dict from a node to a list of edges, containing the end node and the weight
        """

        if any(map(lambda node: node not in self.get_nodes(), starting_nodes)):
            return []

        outcoming_cut: dict[int, list[tuple[int, int]]] = {}

        for node in starting_nodes:
            edges = list(filter(lambda x : x[0] not in starting_nodes, self.edges[node]))
            
            if len(edges) > 0:
                outcoming_cut[node] = edges

        return outcoming_cut
    
    def get_incoming_cut(self, starting_nodes: list[int]) -> dict[int, list[tuple[int, int]]]:
        """
        Computes and lists all edges connecting all other nodes of the graph to the "starting_nodes".
        Output is a dict from a node to a list of edges, containing the end node and the weight
        """

        return self.get_outcoming_cut(list(filter(lambda node: node not in starting_nodes, self.get_nodes())))


    def get_reachable_nodes(self, start_node: int) -> list[int]:
        """
        Lists all nodes that are reachable from the specified one. Implemented through a BFS algorithm.
        """

        if start_node not in self.edges:
            return []

        reachable_nodes = []
        
        nodes = deque()
        nodes.append(start_node)
        
        while len(nodes) > 0:
            node = nodes.pop()
            reachable_nodes.append(node)

            for adjacent in self.get_adjacent_nodes(node):
                if adjacent not in reachable_nodes and adjacent not in nodes:
                    nodes.append(adjacent)

        return sorted(reachable_nodes)
    
    def are_nodes_connected(self, node1: int, node2: int = None) -> bool:
        """
        Returns whether the two nodes are connected, i.e. there is a path connecting them.
        """
        
        return self.get_reachable_nodes(node1, end_node=node2) == [node2]
    
    def is_graph_connected(self) -> bool:
        "Returns whether the graph is connected, i.e. all distincts nodes are connected to each other"
        
        for node1 in self.edges:
            reachable_nodes: list[int] = self.get_reachable_nodes(node1) 

            for node2 in self.edges:
                if node1 != node2 and node2 not in reachable_nodes:
                    return False
                
        return True
            
    def is_graph_complete(self) -> bool:
        """
        Returns whether the graph is complete, i.e. if an edge exists between any two distinct nodes
        """
        
        for node1 in self.edges:
            adjacent_nodes: list[int] = self.get_adjacent_nodes(node1)
            
            for node2 in self.edges:
                if node1 != node2 and node2 not in adjacent_nodes:
                    return False
                
        return True

    @abc.abstractmethod
    def is_graph_acyclic(self) -> bool:
        """
        Returns True if there is no cycle inside the graph, False otherwise
        """
        return 

    def is_subgraph_of(self, other) -> True:
        """
        Returns True if the graph is subgraph of another graph given in input
        """

        if self.__class__ != other.__class__:
            return False
        
        for node in self.edges:
            if node not in other.edges:
                return False
            
            for edge in self.edges[node]:
                if edge not in other.edges[node]:
                    return False
                
        return True
    
    def is_tree(self) -> bool:
        """
        Returns whether the graph is a tree, i.e. it's connected and acyclic
        """

        return self.is_graph_connected() and self.is_graph_acyclic()
    
    def is_spanning_tree_of(self, other) -> bool:
        """
        Returns whether the graph is a spanning tree, i.e. if it is a tree and it contains all nodes\"
        """
        
        for edge in self.edges:
            if edge not in other.edges:
                return False
        
        return self.is_tree()
    
    #def get_spanning_tree(self):
    #    """
    #    Returns a spanning tree of self. Implemented through Prim's algorithm
    #    """
#
    #    start_node: int = [edge[0] for edge in self.edges][random.randint(0, len(self.edges))]
    #    sorted(self.edges[start_node]):




class DirectedGraph(Graph):
    def add_edge(self, node1: int, node2: int, weight: int = 1) -> bool:
        if node1 in self.edges and node2 in self.get_adjacent_nodes(node1):
            return False
        
        if node1 not in self.edges:
            self.add_node(node1)
        
        if node2 not in self.edges:
            self.add_node(node2)

        self.edges[node1].append((node2, weight))
        self.edges[node1] = sorted(self.edges[node1])
        
        return True
    
    def delete_edge(self, node1: int, node2: int) -> bool:
        if node1 not in self.edges or node2 not in self.get_adjacent_nodes(node1):
            return False
        
        for edge in self.edges[node1]:
            if edge[0] == node2:
                self.edges[node1].remove(edge)  # remove the edge
                
                return True
            
        return False    # return false if loop never found the corresponding edge
    
    def is_graph_acyclic(self) -> bool:
        visited = []
        recursion_stack = []

        def leads_to_cycle(graph, node: int):
            visited.append(node)
            recursion_stack.append(node)

            adjacent_nodes = self.get_adjacent_nodes(node)
            
            for neighbor in adjacent_nodes:
                if neighbor not in visited:
                    if leads_to_cycle(graph, neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(node)
            return False

        for node in self.edges:
            if node not in visited:
                if leads_to_cycle(self, node):
                    return False
        
        return True
    
    


class UndirectedGraph(Graph):
    def add_edge(self, node1: int, node2: int, weight: int = 1, self_called = False) -> bool:
        if node1 in self.edges:
            if node2 in self.get_adjacent_nodes(node1):
                return False
            
            self.edges[node1].append((node2, weight))
            self.edges[node1] = sorted(self.edges[node1])
        else:    
            self.edges[node1] = [(node2, weight)]
            
        if node2 not in self.edges:
            self.edges[node2] = []

        if not self_called:
            self.add_edge(node2, node1, weight=weight, self_called=True)

        return True
    
    def delete_edge(self, node1: int, node2: int, self_called: bool = False) -> bool:
        if node1 not in self.edges or node2 not in self.get_adjacent_nodes(node1):
            return False
        
        for edge in self.edges[node1]:
            if edge[0] == node2:
                self.edges[node1].remove(edge)  # remove the edge
                
                if not self_called:
                    self.delete_edge(node2, node1, self_called=True)
                
                return True

        return False
    
    def is_graph_acyclic(self) -> bool:
        def leads_to_cycle(graph, node: int, visited: list[int], parent: int = None):
            visited.append(node)

            adjacent_nodes: list[int] = graph.get_adjacent_nodes(node)
            if parent in adjacent_nodes:
                adjacent_nodes.remove(parent)

            for neighbor in adjacent_nodes:
                if neighbor in visited or leads_to_cycle(graph, neighbor, visited, parent=node):
                    return True

            return False
        
        visited: list[int] = []

        for node in self.edges:
            if node not in visited:
                if leads_to_cycle(self, node, visited):
                    return False
        
        return True