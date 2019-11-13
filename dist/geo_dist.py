import math
import numpy as np
from scipy.spatial.distance import cdist
from utils.transformations import rotation_matrix
from utils.lin_assign import opt_assign

from dist.abstract_dist import Abstract_Dist
from problems import Abstract_Prob


class Geo_Dist(Abstract_Dist):

    _step = None

    def __init__(self, step=0.02):
        super().__init__()

        self._step = step

    def _normalise(self, inst):
        # centering the coords
        coord = np.dot(np.identity(inst.n) - 1.0 / inst.n * np.ones((inst.n, inst.n)), inst)
        # scaling the coords
        coord /= math.sqrt(np.power(coord, 2).sum() / inst.n)

        return coord

    def dist(self, inst_a: Abstract_Prob, inst_b: Abstract_Prob):
        coord_a = self._normalise(inst_a)
        coord_b = self._normalise(inst_b)

        min_dist = None
        min_sum_dist = np.inf

        for alpha in np.arange(0.0, 2 * math.pi, self._step * math.pi):
            M = rotation_matrix(alpha, (0, 0, 1), (0, 0, 0))
            coord_tr = coord_a.dot(M[:2, :2])
            dist_mat = cdist(coord_tr, coord_b)

            sum_dist = dist_mat.min(axis=1).sum()

            if min_sum_dist > sum_dist:
                min_sum_dist = sum_dist
                min_dist = dist_mat

        Pi = None
        # work for optimal assignment
        Pi = opt_assign(min_dist)
        min_sum_dist = np.multiply(min_dist, Pi).sum()

        # self.inst_score = math.exp(-min_sum_dist)
        return {'dist': min_sum_dist, 'pi': np.argmax(Pi, axis=1)}
