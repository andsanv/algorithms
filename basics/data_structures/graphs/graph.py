from collections import deque


class Graph:
    edges: dict[int, list[tuple[int, int]]] # every key is a node, every value is a list of tuples, where the first element is the other node and the second element is the weight
    is_weighted: bool
    weighted_counter: int 
    
    
    def __init__(self):
        self.edges = {}
        self.is_weighted = False
        self.weighted_counter = 0

    
    def add_edge(self, node1: int, node2: int, weight: int = 1, directed: bool = True) -> bool:
        """
        Adds the specified edge to the graph. If directed is set to True, also adds the opposite edge.
        """
        
        if node1 in self.edges:
            if node2 in self.edges[node1]:
                return False
            
            self.edges[node1].append((node2, weight))
            sorted(self.edges[node1])
        else:    
            self.edges[node1] = [(node2, weight)]
            sorted(self.edges)

        if not directed:
            self.add_edge(node2, node1, weight)
        
        if weight > 1:
            self.weighted_counter += 1
            self.is_weighted = True
        
        return True
    

    def delete_edge(self, node1: int, node2: int, directed: bool = True) -> bool:
        """
        Deletes the specified edge from the graph, if it exists.
        If directed is set to True, it also deletes the opposite edge, if it exists.
        """

        if node1 not in self.edges or node2 not in self.get_adjacent_nodes(node1):
            return False
        
        for edge in self.edges[node1]:
            if edge[0] == node2:
                if edge[1] > 1: # decrement weighted edges counter if the one to delete is weighted
                    self.weighted_counter -= 1
                    if self.weighted_counter == 0:
                        self.is_weighted = False    # set graph to not weighted if there is no more weighted edges

                self.edges[node1].remove(edge)  # remove the edge

                if len(self.edges[node1]) == 0: # remove node from dict if it has no more edges
                    self.edges.pop(node1)
                
                if not directed:
                    self.delete_edge(node2, node1, True)    # also delete opposite diretion
                
                return True
            
        return False    # return false if loop never found the corresponding edge


    
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
    

    def get_reachable_nodes(self, start_node: int, end_node: int = None) -> list[int]:
        """
        Lists all nodes that are reachable from the specified one. Implemented through a BFS algorithm.
        "end_node" is an optional and internal parameter to improve speed of the "is_connected()" function.
        """

        if start_node not in self.edges:
            return []
        
        if start_node == end_node:
            return [end_node]

        reachable_nodes = []
        
        nodes = deque()
        nodes.append(start_node)
        
        while len(nodes) > 0:
            node = nodes.pop()
            reachable_nodes.append(node)

            for adjacent in graph.get_adjacent_nodes(node):
                if end_node != None and adjacent == end_node:
                    return [end_node]

                if adjacent not in reachable_nodes and adjacent not in nodes:
                    nodes.append(adjacent)

        return sorted(reachable_nodes) if not end_node else []
    
    def are_nodes_connected(self, node1: int, node2: int = None) -> bool:
        """
        Returns whether the two nodes are connected, i.e. there is a path connecting them.
        """
        
        return self.get_reachable_nodes(node1, end_node=node2) == [node2]
    
    def is_graph_connected(self) -> bool:
        "Returns whether the graph is connected, i.e. all distincts nodes are connected to each other"
        
        for node1 in self.edges:
            for node2 in self.edges:
                if node1 != node2 and not self.are_nodes_connected(node1, node2):
                    print(f"node {node1} and node {node2} are not connected")
                    return False
                
        return True
            
    def is_graph_complete(self) -> bool:
        """
        Returns whether the graph is complete, i.e. if an edge exists between any two distinct nodes
        """
        
        for node1 in self.edges:
            for node2 in self.edges:
                if node1 != node2 and node2 not in graph.get_adjacent_nodes(node1):
                    return False
                
        return True

    def is_graph_acyclic(self) -> bool:
        """
        Returns true if there is no cycle inside the graph, false otherwise
        """

        def is_cyclic(node: int, visited: list[int], current_recursion: list[int]) -> bool:
            """
            Helper for the "is_graph_acyclic" function
            """

            visited.append(node)
            current_recursion.append(node)

            for neighbor in (self.get_adjacent_nodes(node)):
                if neighbor not in visited:
                    if is_cyclic(neighbor, visited, current_recursion):
                        return True
                elif neighbor in current_recursion:
                    return True
                    
            current_recursion.remove(node)
            return False
        
        visited: list[int] = [] # list of nodes already visited
        current_recursion: list[int] = []   # list of nodes in the current recursion branch

        for node in self.edges:
            if not node in visited:
                if is_cyclic(node, visited, current_recursion):
                    return False
        
        return True

