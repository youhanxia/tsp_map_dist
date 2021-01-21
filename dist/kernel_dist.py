import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
# from grakel import GraphKernel, Graph
import igraph as ig
import graphkernels.kernels as gk


from dist import Abstract_Dist
from problem import TSP


class Kernel_Dist(Abstract_Dist):
    kernel = gk.CalculateGeometricRandomWalkKernel

    def __init__(self, kernel=None):
        super().__init__()
        if kernel == 'graphlet_connected':
            self.kernel = gk.CalculateConnectedGraphletKernel
        elif kernel == 'edge_hist_gauss':
            self.kernel = gk.CalculateEdgeHistGaussKernel
        elif kernel == 'edge_hist':
            self.kernel = gk.CalculateEdgeHistKernel
        elif kernel == 'random_walk_exp':
            self.kernel = gk.CalculateExponentialRandomWalkKernel
        elif kernel == 'random_walk_geo':
            self.kernel = gk.CalculateGeometricRandomWalkKernel
        elif kernel == 'graphlet':
            self.kernel = gk.CalculateGraphletKernel
        elif kernel == 'random_walk_step':
            self.kernel = gk.CalculateKStepRandomWalkKernel
        elif kernel == 'shortest_path':
            self.kernel = gk.CalculateShortestPathKernel
        elif kernel == 'vertex_edge_hist_gauss':
            self.kernel = gk.CalculateVertexEdgeHistGaussKernel
        elif kernel == 'vertex_edge_hist':
            self.kernel = gk.CalculateVertexEdgeHistKernel
        elif kernel == 'vertex_hist+gauss':
            self.kernel = gk.CalculateVertexHistGaussKernel
        elif kernel == 'vertex_hist':
            self.kernel = gk.CalculateVertexHistKernel
        elif kernel == 'vertex_vertex_edge_hist':
            self.kernel = gk.CalculateVertexVertexEdgeHistKernel
        elif kernel == 'weisfeiler_lehman':
            self.kernel = gk.CalculateWLKernel

        self.lbl = 'kernel_' + kernel if kernel is not None else 'none'

    def _construct_graph(self, inst: TSP, th=1):
        edge_weights = list(pdist(inst))
        edges = list(combinations(range(inst.n), 2))
        edges = sorted(zip(edges, edge_weights), key=lambda x: x[1])
        edges = edges[:int(len(edges) * th)]

        g = ig.Graph(n=inst.n)
        g.vs['label'] = [0] * n

        for e, w in edges:
            g.add_edge(e[0], e[1], label=w)

        return g.as_undirected()

    def dist(self, inst_a: TSP, inst_b: TSP):
        # return kernel similarity directly
        # not in the form of distances, i.e. not defined in [0, inf)

        thresholds = [
            4 / (inst_a.n - 1),
            2 * np.log2(inst_a.n) / (inst_a.n - 1),
            2 * np.sqrt(inst_a.n) / (inst_a.n - 1),
            0.5,
            1,
        ]
        dists = []
        for th in thresholds:
            g_a = self._construct_graph(inst_a, th=th)
            g_b = self._construct_graph(inst_b, th=th)

            ker_sim = self.kernel([g_a, g_b])
            dists.append(ker_sim[0][1])

        return {'dist': np.mean(dists), 'pi': np.arange(inst_b.n)}

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
kernels = [
    # 'graphlet_connected',
    'edge_hist_gauss',
    'edge_hist',
    'random_walk_exp',
    'random_walk_geo',
    # 'graphlet',
    # 'random_walk_step',
    # 'shortest_path',
    'vertex_edge_hist_gauss',
    'vertex_edge_hist',
    # 'vertex_hist+gauss',
    # 'vertex_hist',
    'vertex_vertex_edge_hist',
    # 'weisfeiler_lehman',
    ]


def main0():
    insts = [TSP(n) for _ in range(10)]
    for k in kernels:
        metric = Kernel_Dist(k)
        print(k)
        for i in range(0, 10, 2):
            res = metric.dist(insts[i], insts[i + 1])
            print('\t', res['dist'])
        print()


if __name__ == '__main__':
    main0()
