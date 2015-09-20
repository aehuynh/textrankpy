import networkx as nx

class PageRankGraph(object):
    """Represents an undirected graph used for PageRank. Each node has a score
    attribute. This class encapsulates the PageRank logic and uses a Networkx
    graph.

    Networkx has a native implementation of PageRank. This was created as a
    learning experience.
    """
    def __init__(self, damping_factor=0.85, convergence_threshold=0.0001, cooccurence_threshold=2):
        self.damping_factor = damping_factor
        self.convergence_threshold = convergence_threshold
        self.cooccurence_threshold = cooccurence_threshold
        self.graph = nx.Graph()

    def add(self, values):
        """Add all values and create edges if a value is within
        cooccurence_threshold of another value.
        """
        prevs = []
        for value in values:
            self.graph.add_node(value, score=1)
            if prevs:
                # Create edge with all nodes in prevs
                for prev in prevs:
                    self.graph.add_edge(value, prev)
            # Remove first node in prevs
            if len(prevs) >= self.cooccurence_threshold:
                prevs.pop(0)

            prevs.append(value)

    def pagerank(self, max_iter=None, data=False):
        """Performs pagerank and returns all nodes sorted in descending order.

        :param data: whether to return a nx.graph node with all its data or
        just the value of the node.
        """
        if max_iter is None:
            max_iter = self.graph.number_of_nodes()

        # Recalculate the node scores max_iter times
        for _in in range(max_iter):
            # Break if converged
            if self.iterate_pagerank():
                break

        # Sort nodes by score
        sorted_nodes = sorted(self.graph.nodes(data=True), key= lambda node: node[1]['score'], reverse=True)

        if not data:
            # Create list with only the values of the nodes
            sorted_nodes = list(map(lambda node: node[0], sorted_nodes))

        return sorted_nodes

    def iterate_pagerank(self):
        """Recalculates score of every node. Returns true if all scores have
        converged.
        """
        converged = True

        for node in self.graph.nodes(data=True):
            old_score = node[1]['score']
            # Calculate and set the new score
            self.graph.add_node(node[0], score=self.calculate_score(node[0]))

            if converged and abs(old_score - node[1]['score']) > self.convergence_threshold:
                converged = False

        return converged

    def calculate_score(self, node):
        """Calculates the score of the node. The equation is copied
        from the article 'TextRank: Bringing Orders':
        http://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf.
        """
        score_from_neighbors = sum(map(lambda n: self.graph.node[n]['score'] / len(self.graph.neighbors(n)), self.graph.neighbors(node)))
        return (1-self.damping_factor) + self.damping_factor * score_from_neighbors
