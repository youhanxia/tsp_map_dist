# import os
# import sys
import numpy as np
from scipy.spatial.distance import euclidean
from concorde.tsp import TSPSolver


# generate a 2d tsp instance with n points
# the minimum of pairwise distance can be constrained by passing in a number to sparse_threshold
# be careful that a large sparse_threshold slows the function significantly even causes a dead loop
def tsp_instance(n, map_size=256, seed=None, sparse_threshold=None):
    if seed is not None:
        np.random.seed(seed)
    if sparse_threshold:
        inst = []
        for i in range(n):
            sparse_violation = True
            p = None
            while sparse_violation:
                sparse_violation = False
                p = np.random.randint(map_size, size=2)
                for q in inst:
                    if euclidean(p, q) < sparse_threshold:
                        sparse_violation = True
            inst.append(p)
        inst = np.array(inst)
    else:
        inst = np.random.randint(map_size, size=(n, 2))
    return inst


# solve with py-concorde
# https://github.com/jvkersch/pyconcorde
# return the optimal tour as an np array followed by its travel cost
def solve(inst):
    # precision factor
    pr = 1e-3

    # # shut output, not working
    # stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')

    # solve
    slr = TSPSolver.from_data(inst[:, 0] / pr, inst[:, 1] / pr, 'EUC_2D')
    res = slr.solve(verbose=False)

    # # redirect output
    # sys.stdout = stdout

    return res.tour, res.optimal_value


# output tsp instance in chunks.
# A file consists of chunk_size lines,
# each of which is an instance as flattened coordinates followed by its optimal tour, delimited by ';'.
# i.e. p0_x, p0_y, p1_x, p1_y, ..., pn_x, pn_y; s0, s1, ..., sn
def generate_dataset(size=200000, n=(20, 100), chunk_size=1000):
    i = 0
    while size:
        if size > chunk_size:
            m = chunk_size
            size -= chunk_size
        else:
            m = size
            size = 0
        with open('tsp_training_vol_' + str(i) + '.csv', 'w') as f:
            for _ in range(m):
                if isinstance(n, int):
                    nn = n
                else:
                    nn = np.random.randint(n[0], n[1] + 1)

                # generate each instance
                inst = tsp_instance(nn)

                # solve the instance
                tour, fit = solve(inst)

                # print in a single line, change if needed
                print(','.join(inst.flatten(order='C').astype(str)), file=f, end=';')
                print(','.join(tour.astype(str)), file=f)
        i += 1


if __name__ == '__main__':
    generate_dataset(size=10)
