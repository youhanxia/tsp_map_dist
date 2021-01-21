import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from problem import TSP
from solver import Abstract_Solver


class DMST_Heuristic(Abstract_Solver):
    lbl = 'heuristic_dbl_mst'

    def solve(self, inst: TSP, eval=None):
        mst = minimum_spanning_tree(inst.dist_mat).toarray()
        mst += mst.T

        tour = [np.random.choice(inst.n)]

        self._dfs(mst, tour)

        tour = np.array(tour)
        return {'tour': tour, 'fitness': inst.eval(tour)}

    def _dfs(self, mst, tour):
        u = tour[-1]
        neighs = np.nonzero(mst[u])[0]
        neighs = set(neighs) - set(tour)
        for v in neighs:
            tour.append(v)
            self._dfs(mst, tour)


