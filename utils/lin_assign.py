import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory


def opt_assign(dist_mat):
    d = len(dist_mat)

    model = ConcreteModel()

    model.i = Set(initialize=range(d))
    model.j = Set(initialize=range(d))

    def dist_rule(model, i, j):
        return dist_mat[i][j]

    model.d = Param(model.i, model.j, initialize=dist_rule)

    model.x = Var(model.i, model.j, bounds=(0, 1))

    def row_rule(model, i):
        return sum(model.x[i, j] for j in model.j) == 1

    model.c1 = Constraint(model.i, rule=row_rule)

    def column_rule(model, j):
        return sum(model.x[i, j] for i in model.i) == 1

    model.c2 = Constraint(model.j, rule=column_rule)

    def objective_rule(model):
        return sum(model.d[i, j] * model.x[i, j] for i in model.i for j in model.j)

    model.objective = Objective(rule=objective_rule, sense=minimize)

    opt = SolverFactory('glpk')
    opt.solve(model)

    model_x = model.x.get_values()

    x = np.zeros((d, d))

    for i, j in model_x.keys():
        x[i, j] = model_x[(i, j)]

    return x


if __name__ == '__main__':
    dist = np.random.random((5, 5))
    print(dist)
    Pi = opt_assign(dist)
    print(opt_assign(dist))
    print(np.argmax(Pi, axis=0))
    print(np.argmax(Pi, axis=1))
