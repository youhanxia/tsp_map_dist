import math
import numpy as np

from problems import TSP
from tour_dist import Abstact_Tour_Dist


class Step_Turn_Dist(Abstact_Tour_Dist):

    def dist(self, inst_a: TSP, inst_b: TSP, tour_a=None, tour_b=None, **kwargs):
        if tour_a is None:
            tour_a = inst_a.optimal_tour
        if tour_b is None:
            tour_b = inst_b.optimal_tour

        # make sure n_b is larger
        if inst_b.n < inst_a.n:
            inst_a, inst_b = inst_b, inst_a
            tour_a, tour_b = tour_b, tour_a

        coords_a = np.array(inst_a)
        coords_b = np.array(inst_b)
        fit_a = inst_a.eval(tour_a)
        fit_b = inst_b.eval(tour_b)

        coords_a = coords_a / np.sqrt(fit_a / inst_a.n)
        coords_b = coords_b / np.sqrt(fit_b / inst_b.n)

        sol_a = self._translate(coords_a, tour_a)
        sol_b = self._translate(coords_b, tour_b)

        best_dist = np.inf
        cost_ab = np.array([[self._cost(sol_a[i], sol_b[j]) for j in range(inst_b.n)] for i in range(inst_a.n)])
        cost_b = np.array([self._cost(None, sol_b[j]) for j in range(inst_b.n)])
        for offset in range(inst_a.n):
            diff = np.zeros((inst_a.n, inst_b.n))
            diff[0][0] = cost_ab[offset][0]
            for j in range(1, inst_b.n - inst_a.n + 1):
                diff[0][j] = min((diff[0][j - 1] + cost_b[j], np.sum(cost_b[:j-1]) + cost_ab[offset][j]))
            for i in range(1, inst_a.n):
                ii = offset + i - inst_a.n
                diff[i][i] = diff[i - 1][i - 1] + cost_ab[ii][ii]
                for j in range(i + 1, inst_b.n - inst_a.n + i + 1):
                    diff[i][j] = min((diff[i][j - 1] + cost_b[j], diff[i - 1][j - 1] + cost_ab[ii][j]))
            if best_dist > diff[-1][-1]:
                best_dist = diff[-1][-1]
        return {'dist': best_dist}

    @staticmethod
    def _cost(step_a, step_b):
        if step_a is None:
            step_a = (0.0, 0.0)

        arc = abs(step_a[0] - step_b[0]) / 180.0 * math.pi * (step_a[1] + step_b[1]) / 2.0
        axis = step_a[1] - step_b[1]

        return math.sqrt(arc ** 2 + axis ** 2)

    @staticmethod
    def _translate(coords, tour):
        n = coords.shape[0]

        angle = []
        cost = []
        for i in range(-2, n - 2):
            # calculate angle of turning
            u = coords[i]
            v = coords[i + 1]
            w = coords[i + 2]
            e_1 = v - u
            e_2 = w - v
            theta = np.arccos(np.dot(e_1, e_2) / (np.linalg.norm(e_1) * np.linalg.norm(e_2)))
            if e_1[0] * e_2[1] - e_1[1] * e_2[0] < 0:
                theta = -theta
            angle.append(theta)
            cost.append(np.linalg.norm(e_1))
        angle = np.array(angle)
        cost = np.array(cost)
        # if clockwise, flip the tour
        if angle.sum() < 0:
            angle = -angle[::-1]
            cost = cost[::-1]
            tour[:] = tour[::-1]
        return list(zip(angle, cost))
