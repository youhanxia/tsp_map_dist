from problems import TSP


class Abstact_Tour_Dist(object):

    def dist(self, inst_a: TSP, inst_b: TSP, tour_a=None, tour_b=None, **kwargs):
        return {'dist': abs(inst_a.n - inst_b.n)}
