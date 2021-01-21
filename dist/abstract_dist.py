from problem import Abstract_Prob
import numpy as np


class Abstract_Dist(object):
    lbl = 'dummy'
    def dist(self, inst_a: Abstract_Prob, inst_b: Abstract_Prob):
        return {'dist': abs(inst_a.n - inst_b.n), 'pi': np.random.permutation(inst_b.n)}
