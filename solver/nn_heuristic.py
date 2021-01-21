import numpy as np
from problem import TSP
from solver import Abstract_Solver


class NN_Heuristic(Abstract_Solver):
    lbl = 'heuristic_near_neigh'

    def solve(self, inst: TSP, eval=None):
        tour = [np.random.choice(inst.n)]
        working_dm = np.array(inst.dist_mat)
        for _ in range(inst.n - 1):
            working_dm[:, tour[-1]] = np.inf
            tour.append(np.argmin(working_dm[tour[-1]]))
        tour = np.array(tour)
        return {'tour': tour, 'fitness': inst.eval(tour)}
