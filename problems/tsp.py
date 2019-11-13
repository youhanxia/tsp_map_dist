import numpy as np
from scipy.spatial.distance import pdist, squareform

from problems import Abstract_Prob


class TSP(np.ndarray, Abstract_Prob):

    n = 0
    dist_mat = None
    optimal_tour = None
    name = 'arbitrary_inst'

    def __new__(cls, *args):
        obj = args[0]
        if type(obj) is np.ndarray and obj.shape[1] == 2:
            # initialise with coordinate array
            return np.asarray(obj).view(cls)
        elif type(obj) is int and obj > 2:
            # randomly initialise with size n
            return (2 * np.random.random((obj, 2)) - 1).view(cls)
        elif type(obj) is dict:
            # initialise with a possibly solved instance as a dict
            return np.asarray(obj['instance']).view(cls)

    def __init__(self, *args):
        obj = args[0]
        if type(obj) is dict:
            if 'solution' in obj:
                self.optimal_tour = obj['solution']
            if 'name' in obj:
                self.name = obj['name']

    def __array_finalize__(self, obj):
        self.n = self.shape[0]
        self.dist_mat = squareform(pdist(self))

    # fitness function can be called from a solver, return the optimal fitness if tour not given
    # optimal fitness not stored
    def eval(self, tour=None):
        if tour is None:
            tour = self.optimal_tour
        fit = 0.0
        for i in range(self.n):
            fit += self.dist_mat[tour[i - 1], tour[i]]
        return [fit]

    def two_opt(self, tour):
        new_tour = np.array(tour)

        for i in range(-1, self.n - 1):
            for j in range(i + 2, self.n - 1):
                a, b, c, d = new_tour[i], new_tour[i + 1], new_tour[j], new_tour[j + 1]
                if self.dist_mat[a][c] + self.dist_mat[d][b] < self.dist_mat[a][b] + self.dist_mat[d][c]:
                    new_tour[i + 1: j + 1] = new_tour[i + 1: j + 1][:: -1]
        return new_tour


if __name__ == '__main__':
    tsp = TSP({'instance': [[1, 1], [2, 2]], 'solution': [0, 1]})
    print(tsp.optimal_tour)
