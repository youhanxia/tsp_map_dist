class Abstract_Prob(object):

    n = 0

    def __init__(self, n):
        self.n = n

    def eval(self, sln=None):
        return [0.0]
