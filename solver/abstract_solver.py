import numpy as np

from problems import Abstract_Prob


class Abstract_Solver(object):

    def solve(self, inst: Abstract_Prob, eval=None):
        tour = np.random.permutation(inst.n)
        return {'tour': tour, 'fitness': inst.eval(tour)}
