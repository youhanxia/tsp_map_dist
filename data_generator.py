import os
# import sys
from string import digits
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from concorde.tsp import TSPSolver
from matplotlib import pyplot as plt

from problem import TSP


def viz(inst, title=None):
    xs = np.array(inst)[:, 0]
    ys = np.array(inst)[:, 1]

    plt.clf()
    plt.title(title)
    plt.plot(xs, ys, '.')
    plt.show()


def random_func(phi):
    def uniform_func(size):
        return np.random.random(size=size) * 2 - 1

    def normal_func(size):
        return np.random.normal(scale=0.2, size=size)

    def exp_func(size):
        return np.random.exponential(scale=0.2, size=size) * np.random.choice([1, -1], size)

    if phi is None or type(phi) is str:
        phi = [phi]

    rf = []
    for t in phi:
        if t == 'normal':
            rf.append(normal_func)
        elif t == 'exp':
            rf.append(exp_func)
        else:
            rf.append(uniform_func)

    if len(rf) < 2:
        rf.append(rf[0])

    return rf


def tsp_instance(n, phi=None):
    rf_x, rf_y = random_func(phi)

    inst = np.zeros((n, 2))
    inst[:, 0] = rf_x(size=n)
    inst[:, 1] = rf_y(size=n)

    while True:
        idx = (cdist(np.zeros((1, 2)), inst) > 1)[0]
        m = np.sum(idx)
        if m == 0:
            return TSP(inst)
        inst[idx, 0] = rf_x(size=m)
        inst[idx, 1] = rf_y(size=m)


def harden_seed(inst_0, max_iter=100, num_candi=10):
    inst = TSP(inst_0)

    for i in range(max_iter):
        inst_new = inst
        fit_new = inst_new.hardness_est()
        for j in range(num_candi):
            inst_x = resample_from_seed(inst, 0.1, 0.5)
            fit_x = inst_x.hardness_est()
            if fit_new > fit_x:
                inst_new = inst_x
                fit_new = fit_x
        print(fit_new)
        inst = inst_new

    return inst


def sub_sample(inst_0, n):
    idx = np.sort(np.random.choice(len(inst_0), n, replace=False))
    return TSP(inst_0[idx])


def resample_from_seed(inst_0, c, std=0.0):
    inst = TSP(inst_0)
    n = len(inst)
    m = int(c * n)
    idx = np.random.choice(n, m, replace=False)

    if std:
        inst[idx] = np.random.normal(inst[idx], scale=std)
        # bounce out of scope points back
        idx = (cdist(np.zeros((1, 2)), inst) > 1)[0]
        points = np.array(inst)[idx]
        norms = np.linalg.norm(points, axis=1)
        inst[idx, 0] = points[:, 0] / (norms ** 2)
        inst[idx, 1] = points[:, 1] / (norms ** 2)
    else:
        inst[idx] = tsp_instance(m)
    return inst


def travel_from_seed(inst_0, c):
    inst = TSP(inst_0)
    n = len(inst)
    r = np.mean(cdist(np.zeros((1, 2)), inst)) / np.sqrt(n) * c
    alpha = np.random.random(n) * np.pi * 2
    delta = np.zeros((n, 2))
    delta[:, 0] = np.sin(alpha) * r
    delta[:, 1] = np.cos(alpha) * r

    inst += delta
    return inst


def solve(inst):
    # precision factor
    pr = 1
    if np.mean(np.abs(np.array(inst))) < 100:
        pr = 1e-3

    # solve
    inst = np.array(inst)
    slr = TSPSolver.from_data(inst[:, 0] / pr, inst[:, 1] / pr, 'EUC_2D')
    res = slr.solve(verbose=False)

    return res.tour, res.optimal_value


def read_inst(fn, dir=None):
    if dir is not None:
        fn = os.path.join(dir, fn)

    points = []
    with open(fn) as f:
        for line in f:
            point = line.rstrip().split()
            points.append(point)

    return TSP(np.array(points, dtype=float))


def generate_dataset(size=50, repeat=1, max_mod=0.3):
    data_dir = 'data'

    fns = os.listdir(data_dir)

    for fn in fns:
        if fn.endswith('.pkl') or fn.endswith('.csv') or fn.startswith('.'):
            continue
        out_fn = fn + '_mod_' + str(size) + '.pkl'
        if out_fn in fns:
            continue
        print(fn)
        probs = {}
        seed = read_inst(fn, data_dir)
        if len(seed) > size:
            seed = sub_sample(seed, size)

        probs[fn] = {'instance': np.array(seed), 'solution': solve(seed)[0]}

        for c in range(1, 10, 1):
            for i in range(repeat):
                inst = resample_from_seed(seed, c / 10 * max_mod)
                probs[fn + '_r' + str(c) + '_#' + str(i)] = {'instance': np.array(inst), 'solution': solve(inst)[0]}
                inst = travel_from_seed(seed, c / 10 * max_mod)
                probs[fn + '_t' + str(c) + '_#' + str(i)] = {'instance': np.array(inst), 'solution': solve(inst)[0]}

        with open(os.path.join(data_dir, out_fn), 'wb') as f:
            pickle.dump(probs, f, protocol=pickle.HIGHEST_PROTOCOL)


tl_circuit_prefix = [
    'a',
    'd',
    'dl',
    'fl',
    'linhp',
    'p',
    'pcb',
    'pla',
    'pr',
    'rl',
    'u'
]
tl_city_prefix = [
    'ali',
    'att',
    'berlin',
    'bier',
    'brd',
    'burma',
    'fnl',
    'gr',
    'nrw',
    'ulysses',
    'usa',
]
tl_other_prefix = [
    'ch',
    'dsj',
    'eil',
    'gil',
    'kroA',
    'kroB',
    'kroC',
    'kroD',
    'kroE',
    'rat',
    'rd',
    'st',
    'ts',
    'tsp',
    'vm'
]


def crop(rank):
    idx = set(rank[0])
    if len(idx) <= 50:
        return idx
    while True:
        for i in [0, 1]:
            for j in [0, -1]:
                idx.discard(rank[i].pop(j))
                if len(idx) <= 50:
                    return np.array(list(idx), dtype=int)


def generate_seed(size=50, volume=25):
    data_dir = 'data'
    tl_dir = 'tl_data'

    tl_fns = os.listdir(tl_dir)

    city_fns = []
    circuit_fns = []
    remove_digits = str.maketrans('', '', digits)

    for fn in tl_fns:
        prefix = fn.translate(remove_digits)
        if prefix in tl_circuit_prefix:
            circuit_fns.append(fn)
        elif prefix in tl_city_prefix:
            city_fns.append(fn)

    print(len(city_fns), 'city TSP instances')
    print(len(circuit_fns), 'circuit TSP instances')

    # uniform
    for i in range(volume):
        fn = 'uniform_' + str(i)
        tsp = np.array(tsp_instance(size))
        with open(os.path.join(data_dir, fn), 'w') as f:
            for p in tsp:
                print(p[0], p[1], file=f)

    # exponential
    for i in range(volume):
        fn = 'gaussian_' + str(i)
        tsp = np.array(tsp_instance(size, phi='exp'))
        # tsp[:size//5] = tsp_instance(size//5)
        with open(os.path.join(data_dir, fn), 'w') as f:
            for p in tsp:
                print(p[0], p[1], file=f)

    # cities
    i = 0
    for fn in city_fns:
        tsp = np.array(read_inst(fn, tl_dir))
        if len(tsp) < 50:
            continue
        if len(tsp) < 500:
            ofn = 'city_' + str(i) + '_' + fn
            i += 1
            rank_x = list(np.argsort(tsp[:, 0]))
            rank_y = list(np.argsort(tsp[:, 1]))
            tsp = tsp[crop([rank_x, rank_y])]
            with open(os.path.join(data_dir, ofn), 'w') as f:
                for p in tsp:
                    print(p[0], p[1], file=f)
        elif len(tsp) < 10000:
            rank_x = list(np.argsort(tsp[:, 0]))
            rank_y = list(np.argsort(tsp[:, 1]))

            n = len(rank_x) // 2

            lfn = 'city_' + str(i) + '_' + fn + '_l'
            i += 1
            rank_xl = rank_x[:n]
            rank_yl = list(filter(lambda e: e in rank_xl, rank_y))
            tsp_l = tsp[crop([rank_xl, rank_yl])]
            with open(os.path.join(data_dir, lfn), 'w') as f:
                for p in tsp_l:
                    print(p[0], p[1], file=f)

            rfn = 'city_' + str(i) + '_' + fn + '_r'
            i += 1
            rank_xr = rank_x[n:]
            rank_yr = list(filter(lambda e: e in rank_xr, rank_y))
            tsp_r = tsp[crop([rank_xr, rank_yr])]
            with open(os.path.join(data_dir, rfn), 'w') as f:
                for p in tsp_r:
                    print(p[0], p[1], file=f)
        else:
            rank_x = list(np.argsort(tsp[:, 0]))
            rank_y = list(np.argsort(tsp[:, 1]))

            n = len(rank_x) // 2

            rank_xl = rank_x[:n]
            rank_yl = list(filter(lambda e: e in rank_xl, rank_y))
            rank_xr = rank_x[n:]
            rank_yr = list(filter(lambda e: e in rank_xr, rank_y))

            m = len(rank_xl) // 2

            llfn = 'city_' + str(i) + '_' + fn + '_ll'
            i += 1
            rank_yll = rank_yl[:m]
            rank_xll = list(filter(lambda e: e in rank_yll, rank_xl))
            tsp_ll = tsp[crop([rank_xll, rank_yll])]
            with open(os.path.join(data_dir, llfn), 'w') as f:
                for p in tsp_ll:
                    print(p[0], p[1], file=f)

            ulfn = 'city_' + str(i) + '_' + fn + '_ul'
            i += 1
            rank_yul = rank_yl[m:]
            rank_xul = list(filter(lambda e: e in rank_yul, rank_xl))
            tsp_ul = tsp[crop([rank_xul, rank_yul])]
            with open(os.path.join(data_dir, ulfn), 'w') as f:
                for p in tsp_ul:
                    print(p[0], p[1], file=f)

            m = len(rank_xl) // 2

            lrfn = 'city_' + str(i) + '_' + fn + '_lr'
            i += 1
            rank_ylr = rank_yr[:m]
            rank_xlr = list(filter(lambda e: e in rank_ylr, rank_xr))
            tsp_lr = tsp[crop([rank_xlr, rank_ylr])]
            with open(os.path.join(data_dir, lrfn), 'w') as f:
                for p in tsp_lr:
                    print(p[0], p[1], file=f)

            urfn = 'city_' + str(i) + '_' + fn + '_ur'
            i += 1
            rank_yur = rank_yr[m:]
            rank_xur = list(filter(lambda e: e in rank_yur, rank_xr))
            tsp_ur = tsp[crop([rank_xur, rank_yur])]
            with open(os.path.join(data_dir, urfn), 'w') as f:
                for p in tsp_ur:
                    print(p[0], p[1], file=f)

    # circuits
    i = 0
    for fn in circuit_fns:
        tsp = np.array(read_inst(fn, tl_dir))
        if len(tsp) > 1300:
            continue
        ofn = 'circuit_' + str(i) + '_' + fn
        i += 1
        rank_x = list(np.argsort(tsp[:, 0]))
        rank_y = list(np.argsort(tsp[:, 1]))
        tsp = tsp[crop([rank_x, rank_y])]
        with open(os.path.join(data_dir, ofn), 'w') as f:
            for p in tsp:
                print(p[0], p[1], file=f)


if __name__ == '__main__':
    # generate_dataset(size=10)
    # generate_dataset(size=20)
    # generate_dataset(size=30)
    generate_dataset(size=50)
    # generate_seed(size=50)
