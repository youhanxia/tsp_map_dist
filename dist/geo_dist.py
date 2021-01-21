import os
import sys
import math
import numpy as np
from scipy.spatial.distance import cdist
from utils.transformations import rotation_matrix
from utils.lin_assign import opt_assign
from ge import DeepWalk, Node2Vec, LINE, SDNE, Struc2Vec
import networkx as nx

from dist.abstract_dist import Abstract_Dist
from problem import TSP


class Geo_Dist(Abstract_Dist):
    lbl = 'geometric'

    # _step = None
    # _embedding = None

    def __init__(self, step=0.02, embedding=None):
        super().__init__()

        self._step = step
        self._embedding = embedding
        if embedding is not None:
            self.lbl = embedding

    def dist(self, inst_a: TSP, inst_b: TSP):
        if self._embedding is None:
            coord_a = self._normalise(inst_a)
            coord_b = self._normalise(inst_b)

            min_dist = cdist(coord_a, coord_b)

            min_sum_dist = np.inf
            for alpha in np.arange(0.0, 2 * math.pi, self._step * math.pi):
                M = rotation_matrix(alpha, (0, 0, 1), (0, 0, 0))
                coord_tr = coord_a.dot(M[:2, :2])
                dist_mat = cdist(coord_tr, coord_b)

                sum_dist = dist_mat.min(axis=1).sum()

                if min_sum_dist > sum_dist:
                    min_sum_dist = sum_dist
                    min_dist = dist_mat

        else:
            thresholds = [
                4 / (inst_a.n - 1),
                2 * np.log2(inst_a.n) / (inst_a.n - 1),
                2 * np.sqrt(inst_a.n) / (inst_a.n - 1),
                0.5,
                1,
            ]

            min_dist = np.zeros((inst_a.n, inst_b.n))

            for th in thresholds:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                f = open(os.devnull, 'w')
                sys.stdout = f
                sys.stderr = f

                coord_a = self.get_embeddings(inst_a, th)
                coord_b = self.get_embeddings(inst_b, th)

                f.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

                min_dist += cdist(coord_a, coord_b)

            min_dist /= len(thresholds)

        Pi = None
        # work for optimal assignment
        Pi = opt_assign(min_dist)
        min_sum_dist = np.multiply(min_dist, Pi).sum()

        # self.inst_score = math.exp(-min_sum_dist)
        return {'dist': min_sum_dist, 'pi': np.argmax(Pi, axis=1)}

    def _normalise(self, inst):
        # centering the coords
        coord = np.dot(np.identity(inst.n) - 1.0 / inst.n * np.ones((inst.n, inst.n)), inst)
        # scaling the coords
        coord /= math.sqrt(np.power(coord, 2).sum() / inst.n)

        return coord

    def _compose_edge_list(self, d_mat, th=1):
        n = d_mat.shape[0]
        edges = []
        for i in range(n):
            for j in range(n):
                edges.append([i, j, d_mat[i][j]])

        edges = sorted(edges, key=lambda x: x[2])
        edges = edges[:int(len(edges) * th)]
        edges = [' '.join([str(x) for x in e]) for e in edges]
        return edges

    def get_embeddings(self, inst, th=1):
        G = nx.parse_edgelist(self._compose_edge_list(inst.dist_mat, th), create_using=nx.DiGraph(), nodetype=None,
                                data=[('weight', float)])
        if self._embedding == 'deepwalk':
            model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
            model.train(window_size=5, iter=3)
        elif self._embedding == 'node2vec':
            model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=1)  # init model
            model.train(window_size=5, iter=3)  # train model
        elif self._embedding == 'line':
            model = LINE(G, embedding_size=128, order='second')  # init model,order can be ['first','second','all']
            model.train(batch_size=1024, epochs=50, verbose=2)  # train model
        elif self._embedding == 'sdne':
            model = SDNE(G, hidden_size=[256, 128])  # init model
            model.train(batch_size=3000, epochs=40, verbose=2)  # train model
        elif self._embedding == 'struc2vec':
            model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )  # init model
            model.train(window_size=5, iter=3)  # train model
        else:
            return self._normalise(inst)

        ebds = model.get_embeddings()
        coords = []
        for i in range(inst.n):
            coords.append(ebds[str(i)])
        return np.array(coords)


if __name__ == '__main__':
    # metric = Geo_Dist(embedding='deepwalk')
    metric = Geo_Dist()
    inst_a = TSP(10)
    inst_b = TSP(10)
    print(metric.dist(inst_a, inst_b))
