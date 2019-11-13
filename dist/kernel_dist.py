import numpy as np
from scipy.spatial.distance import pdist, squareform
from grakel import kernels
from grakel import Graph

from dist import Abstract_Dist
from problems import Abstract_Prob, TSP


class Kernel_Dist(Abstract_Dist):

    def dist(self, inst_a: Abstract_Prob, inst_b: Abstract_Prob, kernel='MultiscaleLaplacianFast'):

        g_a = Graph(squareform(pdist(inst_a)))
        g_b = Graph(squareform(pdist(inst_b)))

        # add node labels
        # for some unweighted kernels, construct graph

        model = getattr(kernels, kernel)()

        model = model.fit(g_a)

        dist = model.transform(g_b)

        return {'dist': dist}


n = 10


def main0():
    ia = TSP(n)
    ib = TSP(n)
    metric = Kernel_Dist()
    res = metric.dist(ia, ib)

    print(res['dist'])


if __name__ == '__main__':
    main0()
