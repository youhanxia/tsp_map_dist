import numpy as np
from problem import TSP
from solver import Abstract_Solver


class Gr_Heuristic(Abstract_Solver):
    lbl = 'heuristic_greedy'

    def solve(self, inst: TSP, eval=None):
        args = np.argsort(inst.dist_mat.flatten())[inst.n:]

        n_colour = 0
        colour = np.zeros(inst.n, dtype=int) - 1
        neigh = np.zeros((inst.n, 2), dtype=int) - 1

        for arg in args:
            u = arg // inst.n
            v = arg % inst.n
            if colour[u] < 0 and colour[v] < 0:
                neigh[u][0] = v
                neigh[v][0] = u
                colour[u] = n_colour
                colour[v] = n_colour
                n_colour += 1
            elif colour[u] < 0 and neigh[v][1] < 0:
                neigh[u][0] = v
                neigh[v][1] = u
                colour[u] = colour[v]
            elif colour[v] < 0 and neigh[u][1] < 0:
                neigh[v][0] = u
                neigh[u][1] = v
                colour[v] = colour[u]
            elif colour[u] != colour[v] and neigh[u][1] < 0 and neigh[v][1] < 0:
                neigh[u][1] = v
                neigh[v][1] = u
                np.place(colour, colour == colour[u], colour[v])

        u, v = filter(lambda x: neigh[x][1] == -1, range(inst.n))
        neigh[u][1] = v
        neigh[v][1] = u

        tour = []
        in_tour = [False] * inst.n
        tour.append(0)
        in_tour[0] = True
        while True:
            u = tour[-1]
            v = neigh[u][0]
            if not in_tour[v]:
                tour.append(v)
                in_tour[v] = True
                continue
            v = neigh[u][1]
            if not in_tour[v]:
                tour.append(v)
                in_tour[v] = True
                continue
            break

        tour = np.array(tour)
        return {'tour': tour, 'fitness': inst.eval(tour)}


if __name__ == '__main__':
    tsp = TSP(10)
    slr = Gr_Heuristic()
    sol = slr.solve(tsp)
