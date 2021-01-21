import numpy as np
from problem import TSP
from solver import Abstract_Solver


class FI_Heuristic(Abstract_Solver):
    lbl = 'heuristic_far_ins'

    def solve(self, inst: TSP, eval=None):
        working_dm = np.array(inst.dist_mat)
        working_dm[np.identity(inst.n, dtype=bool)] = -np.inf
        u = np.argmax(np.max(working_dm, axis=1))
        v = np.argmax(working_dm[u])
        tour = [u, v]
        working_dm[:, u] = -np.inf
        working_dm[:, v] = -np.inf
        for _ in range(inst.n - 2):
            u = np.argmax(np.max(working_dm[tour], axis=1))
            v = np.argmax(working_dm[u])
            ind = -1
            min_inc = np.inf
            for i in range(len(tour)):
                inc = inst.dist_mat[v, tour[i]] + inst.dist_mat[tour[i - 1], v] - inst.dist_mat[tour[i - 1], tour[i]]
                if min_inc > inc:
                    min_inc = inc
                    ind = i
            tour.insert(ind, v)
            working_dm[:, v] = -np.inf
        tour = np.array(tour)
        return {'tour': tour, 'fitness': inst.eval(tour)}
