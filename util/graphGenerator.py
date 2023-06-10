import random
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy

#

FOLDER = 'img/'

@dataclass
class GraphGenerator:
    N: int
    p: float
    G: nx.Graph

    def __init__(self, N: int, E: int, rnd_range: int = 100, seed=42):
        random.seed(seed)
        self.N = N
        self.E = E
        # generate a random graph of N nodes and E edges
        self.G = nx.dense_gnm_random_graph(N, E)
        self.G = self.G.to_directed()
        # add random weights to its edges
        for (u, v) in self.G.edges():
            self.G.edges[u, v]['weight'] = random.randint(0, rnd_range)

    # def gnp_random_connected_graph(self):
    #     """
    #     Generates a random undirected graph, similarly to an Erdős-Rényi
    #     graph, but enforcing that the resulting graph is conneted
    #
    #     From https://stackoverflow.com/questions/61958360/
    #     how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx
    #     """
    #     edges = combinations(range(self.N), 2)
    #     G = nx.Graph()
    #     G.add_nodes_from(range(self.N))
    #     if self.p <= 0:
    #         return G
    #     if self.p >= 1:
    #         return nx.complete_graph(self.N, create_using=G)
    #     for _, node_edges in groupby(edges, key=lambda x: x[0]):
    #         node_edges = list(node_edges)
    #         random_edge = random.choice(node_edges)
    #         G.add_edge(*random_edge)
    #         for e in node_edges:
    #             if random.random() < self.p:
    #                 G.add_edge(*e)
    #     # G = nx.fast_gnp_random_graph(self.N, self.p, directed=True, seed=420)
    #     return G

    def get_weights(self):
        """
        Returns the specific weights
        over all the graph edges.

        From https://stackoverflow.com/questions/62564983/
        how-to-efficiently-get-edge-weights-from-a-networkx-graph-nx-graph
        """
        edges = self.G.edges(data=True)
        return dict((x[:-1], x[-1]['weight']) for x in edges if 'weight' in x[-1])

    def save_graph_image(self, s, t):
        """
        Save graph before and after finding the shortest path.

        From https://stackoverflow.com/questions/22785849/
        drawing-multiple-edges-between-two-nodes-with-networkx
        """
        # solve Dijksta on original graph
        path = nx.dijkstra_path(self.G, s, t)
        # Plot nodes
        pos = nx.nx_agraph.graphviz_layout(self.G)
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(self.G, pos, ax=ax)
        nx.draw_networkx_labels(self.G, pos, ax=ax)
        # Plot edges
        curved_edges = [edge for edge in self.G.edges() if reversed(edge) in self.G.edges()]
        straight_edges = list(set(self.G.edges()) - set(curved_edges))
        curved_edges_red = [edge for edge in list(zip(path[:-1], path[1:])) if reversed(edge) in self.G.edges()]
        straight_edges_red = list(set(list(zip(path[:-1], path[1:]))) - set(curved_edges))
        nx.draw_networkx_edges(self.G, pos, ax=ax, edgelist=straight_edges)
        arc_rad = 0.25
        nx.draw_networkx_edges(self.G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
        # Plot labels on edges
        import my_networkx as my_nx
        edge_weights = nx.get_edge_attributes(self.G, 'weight')
        curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
        my_nx.my_draw_networkx_edge_labels(self.G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False,
                                           rad=arc_rad)
        nx.draw_networkx_edge_labels(self.G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)
        fig.savefig(f'{FOLDER}{self.N}_{s}_{t}_original.png',
                    bbox_inches='tight', pad_inches=0)
        # Add the shortest path in red
        nx.draw_networkx_edges(self.G, pos, ax=ax, edgelist=curved_edges_red, connectionstyle=f'arc3, rad = {arc_rad}',
                               edge_color='r')
        nx.draw_networkx_edges(self.G, pos, ax=ax, edgelist=straight_edges_red, edge_color='r')
        fig.savefig(f'{FOLDER}{self.N}_{s}_{t}_color_.png', bbox_inches='tight',
                    pad_inches=0)

    def get_A_matrix(self):
        matrix = nx.incidence_matrix(self.G, oriented=True)
        A = scipy.sparse.csr_matrix.toarray(matrix)
        return A

    def generate_random_graph(self):
        e = self.G.number_of_edges()
        c = np.asarray(list(self.get_weights().values()))
        # Their standard is the other way around.
        A = self.get_A_matrix() * (-1)
        return e, c, A

    def get_longest_path(self):
        longest_path = []
        length_longest_path = -1
        nodes = list(self.G.nodes)
        for s in range(len(nodes)):
            for t in range(len(nodes)):
                try:
                    path_length = nx.shortest_path_length(self.G, s, t, weight='weight')
                    if path_length > length_longest_path:
                        length_longest_path = path_length
                        longest_path = nx.shortest_path(self.G, s, t, weight='weight')
                except:
                    continue
        return longest_path
