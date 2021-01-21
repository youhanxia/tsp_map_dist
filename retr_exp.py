import os
import pickle
import random
import numpy as np
from problem import TSP
from dist import *
from solver import *


def retr_exp_main(data_dir):
    metrics = [
        Map_Dist(),
        Geo_Dist(),
        Geo_Dist(embedding='deepwalk'),
        Geo_Dist(embedding='node2vec'),
        # Geo_Dist(embedding='line'),
        # Geo_Dist(embedding='sdne'),
        Geo_Dist(embedding='struc2vec'),
        Abstract_Dist(),
        Kernel_Dist('edge_hist_gauss'),
        Kernel_Dist('edge_hist'),
        Kernel_Dist('random_walk_exp'),
        Kernel_Dist('random_walk_geo'),
        Kernel_Dist('vertex_edge_hist_gauss'),
        Kernel_Dist('vertex_edge_hist'),
        Kernel_Dist('vertex_vertex_edge_hist'),
        DMST_Heuristic(),
        FI_Heuristic(),
        # Gr_Heuristic(),
        NI_Heuristic(),
        NN_Heuristic()
        ]
    solver = GA_Solver()
    ofn = 'retr_0.txt'

    with open(ofn, 'w') as f:
        print('query metric candidate dist fitness', file=f)

    fns_ = os.listdir(data_dir)
    fns = list(filter(lambda s: s.endswith('.pkl'), fns_))

    data = dict()

    for fn in fns:
        with open(os.path.join(data_dir, fn), 'rb') as f:
            data_ = pickle.load(f)
        data.update(data_)
    keys = list(data.keys())

    queries = list(filter(lambda k: k[-2] != '#', keys))
    random.shuffle(queries)

    for q in queries:
        # candidates = list(filter(lambda k: k.startswith(q), keys))
        # candidates.remove(q)
        # candidates.extend(queries)
        # candidates.remove(q)
        candidates = list(queries)
        candidates.remove(q)

        inst_b = TSP(data[q])
        for m in metrics:
            print(q, m.lbl)
            if m.lbl.startswith('heuristic'):
                fit = solver.solve(inst_b, seed=m.solve(inst_b)['tour'])['fitness']
                with open(ofn, 'a') as f:
                    print(q, m.lbl, None, None, fit, file=f)
                continue
            seeds = []
            for c in candidates:
                inst_a = TSP(data[c])
                d = m.dist(inst_a, inst_b)
                val = d['dist']
                if m.lbl.startswith('kernel'):
                    val = 1.0 / val if val != 0.0 else np.inf
                seed = d['pi'][inst_a.optimal_tour]
                seeds.append((val, seed))
                fit = solver.solve(inst_b, seed=seed)['fitness']
                with open(ofn, 'a') as f:
                    print(q, m.lbl, c, val, fit, file=f)
            if m.lbl == 'mapping':
                seeds = [s[1] for s in sorted(seeds, key=lambda x: x[0])]
                for k in [3, 5, 10]:
                    fit = solver.solve(inst_b, seed=seeds)['fitness']
                    with open(ofn, 'a') as f:
                        print(q, m.lbl, 'top_k', k, fit, file=f)


if __name__ == '__main__':
    retr_exp_main('data')
