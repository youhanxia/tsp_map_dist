import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
# from grakel import GraphKernel, Graph
import igraph as ig
import graphkernels.kernels as gk


from dist import Abstract_Dist
from problems import TSP


class Kernel_Dist(Abstract_Dist):

    def _construct_graph(self, inst: TSP):
        edge_weights = list(pdist(inst))
        edges = list(combinations(range(inst.n), 2))

        g = ig.Graph(n=inst.n)
        g.vs['label'] = [0] * n

        for e, w in zip(edges, edge_weights):
            g.add_edge(e[0], e[1], label=w)

        return g.as_undirected()

    def dist(self, inst_a: TSP, inst_b: TSP, kernel='shortest_path'):
        # return kernel similarity directly
        # not in the form of distances, i.e. not defined in [0, inf)
        g_a = self._construct_graph(inst_a)
        g_b = self._construct_graph(inst_b)

        if kernel == 'random_walk':
            ker_sim = gk.CalculateGeometricRandomWalkKernel([g_a, g_b])
        elif kernel == 'shortest_path':
            ker_sim = gk.CalculateShortestPathKernel([g_a, g_b])
        elif kernel == 'weisfeiler_lehman':
            ker_sim = gk.CalculateWLKernel([g_a, g_b])

        return {'dist': ker_sim[0][1]}

    # def dist(self, inst_a: Abstract_Prob, inst_b: Abstract_Prob, kernel='shortest_path'):
    #     # deprecated GraKel version
    #     g_a = Graph(squareform(pdist(inst_a)), node_labels=list(np.zeros(inst_a.n, dtype=int)))
    #     g_b = Graph(squareform(pdist(inst_b)), node_labels=list(np.zeros(inst_a.n, dtype=int)))
    #
    #     # add node labels
    #     # for some unweighted kernels, construct graph
    #
    #     model = GraphKernel(kernel=kernel)
    #
    #     model.fit_transform([g_a])
    #
    #     dist = model.transform([g_b])
    #
    #     return {'dist': dist[0][0]}


n = 10


def main0():
    ia = TSP(n)
    ib = TSP(n)
    metric = Kernel_Dist()
    res = metric.dist(ia, ib)

    print(res['dist'])


if __name__ == '__main__':
    main0()
