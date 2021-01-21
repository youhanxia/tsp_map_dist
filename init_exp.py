import os
import pickle
import numpy as np
from problem import TSP
from dist import *
from solver import *


def tune_init_exp_main(data_dir):
    metrics = [Map_Dist(sample_size=100), Map_Dist(sample_size=150), Map_Dist(sample_size=200), Map_Dist(sample_size=250), Map_Dist(sample_size=300)]
    lbls = [100, 150, 200, 250, 300]
    solver = GA_Solver()
    ofn = 'tune_X.txt'

    with open(ofn, 'w') as f:
        print('name best', file=f, end=' ')
        for l in lbls:
            print(l, file=f, end=' ')
        print(file=f)

    fns_ = os.listdir(data_dir)
    fns = list(filter(lambda s: s.endswith('.pkl'), fns_))

    for fn in fns:
        with open(os.path.join(data_dir, fn), 'rb') as f:
            data = pickle.load(f)
        keys = list(data.keys())
        i = np.argmin([len(k) for k in keys])
        inst_a = TSP(data[keys[i]])
        inst_b = TSP(data[keys[i] + '_r9_#0'])
        res = []
        print(keys[i])
        for m in metrics:
            d = m.dist(inst_a, inst_b)
            res.append(solver.solve(inst_b, seed=d['pi'][inst_a.optimal_tour])['fitness'])

        with open(ofn, 'a') as f:
            print(keys[i], inst_b.eval(inst_b.optimal_tour)[0], file=f, end=' ')
            for r in res:
                print(r, file=f, end=' ')
            print(file=f)


def init_exp_main(data_dir):
    metrics = [
        Map_Dist(),
        Geo_Dist(),
        Geo_Dist(embedding='deepwalk'),
        Geo_Dist(embedding='node2vec'),
        # Geo_Dist(embedding='line'),
        # Geo_Dist(embedding='sdne'),
        Geo_Dist(embedding='struc2vec'),
        Abstract_Dist(),
        DMST_Heuristic(),
        FI_Heuristic(),
        # Gr_Heuristic(),
        NI_Heuristic(),
        NN_Heuristic()
        ]
    solver = GA_Solver()
    ofn = 'init_0.txt'

    with open(ofn, 'w') as f:
        print('inst_a inst_b best', file=f, end=' ')
        for l in [m.lbl for m in metrics]:
            print(l, file=f, end=' ')
        print(file=f)

    fns_ = os.listdir(data_dir)
    fns = list(filter(lambda s: s.endswith('.pkl'), fns_))

    for fn in fns:
        with open(os.path.join(data_dir, fn), 'rb') as f:
            data = pickle.load(f)
        keys = list(data.keys())
        i = np.argmin([len(k) for k in keys])
        inst_a = TSP(data[keys[i]])
        print(keys[i])
        for k in keys:
            inst_b = TSP(data[k])
            res = []
            for m in metrics:
                if m.lbl.startswith('heuristic'):
                    res.append(solver.solve(inst_b, seed=m.solve(inst_b)['tour'])['fitness'])
                else:
                    d = m.dist(inst_a, inst_b)
                    res.append(solver.solve(inst_b, seed=d['pi'][inst_a.optimal_tour])['fitness'])

            with open(ofn, 'a') as f:
                print(keys[i], k, inst_b.eval(inst_b.optimal_tour)[0], file=f, end=' ')
                for r in res:
                    print(r, file=f, end=' ')
                print(file=f)


if __name__ == '__main__':
    # tune_init_exp_main('data')
    init_exp_main('data')
