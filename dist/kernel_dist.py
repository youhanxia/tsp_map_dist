import numpy as np
from scipy.spatial.distance import pdist, squareform
from grakel import GraphKernel, Graph

from dist import Abstract_Dist
from problems import Abstract_Prob, TSP


class Kernel_Dist(Abstract_Dist):

    def dist(self, inst_a: Abstract_Prob, inst_b: Abstract_Prob, kernel='shortest_path'):

        g_a = Graph(squareform(pdist(inst_a)), node_labels=list(np.zeros(inst_a.n, dtype=int)))
        g_b = Graph(squareform(pdist(inst_b)), node_labels=list(np.zeros(inst_a.n, dtype=int)))

        # add node labels
        # for some unweighted kernels, construct graph

        model = GraphKernel(kernel=kernel)

        model.fit_transform([g_a])

        dist = model.transform([g_b])

        return {'dist': dist[0][0]}


n = 10


def main0():
    ia = TSP(n)
    ib = TSP(n)
    metric = Kernel_Dist()
    res = metric.dist(ia, ib)

    print(res['dist'])


if __name__ == '__main__':
    main0()
