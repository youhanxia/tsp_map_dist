from problem import Abstract_Prob


class Abstract_Dist(object):
    def dist(self, inst_a: Abstract_Prob, inst_b: Abstract_Prob):
        return {'dist': abs(inst_a.n - inst_b.n)}
