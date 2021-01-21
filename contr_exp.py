import os
import pickle
import random
from problem import TSP
from dist import *


def contr_exp_main(data_dir):
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
        ]
    ofn = 'contr_0.txt'

    with open(ofn, 'w') as f:
        print('inst_a inst_b inst_c', file=f, end=' ')
        for l in [m.lbl for m in metrics]:
            print(l, file=f, end=' ')
        print(file=f)

    fns_ = os.listdir(data_dir)
    fns = list(filter(lambda s: s.endswith('.pkl'), fns_))

    data = dict()

    for fn in fns:
        with open(os.path.join(data_dir, fn), 'rb') as f:
            data_ = pickle.load(f)
        data.update(data_)
    keys = list(data.keys())
    # print(keys)

    for _ in range(30):
        for cat in ['circuit', 'city', 'gaussian', 'uniform']:
            ks = list()
            ks.append(random.choice([k for k in keys if k.startswith(cat)]))
            ks.append(random.choice([k for k in keys if k.startswith(cat) and k.split('_')[1] == ks[0].split('_')[1]]))
            ks.append(random.choice([k for k in keys if k.startswith(cat) and k.split('_')[1] != ks[0].split('_')[1]]))
            ks.append(random.choice([k for k in keys if not k.startswith(cat)]))
            print(ks)
            insts = [TSP(data[k]) for k in ks]
            for triples in [[(0, 1), (0, 3), (1, 3)], [(0, 2), (0, 3), (2, 3)]]:
                res = []
                for m in metrics:
                    for i, j in triples:
                        if m.lbl.startswith('kernel'):
                            res.append(1.0 / m.dist(insts[i], insts[j])['dist'])
                        else:
                            res.append(m.dist(insts[i], insts[j])['dist'])
                with open(ofn, 'a') as f:
                    for i, k in enumerate(ks):
                        if (0, 1) in triples and i == 2:
                            continue
                        if (0, 2) in triples and i == 1:
                            continue
                        print(k, file=f, end=' ')
                    for r in res:
                        print(r, file=f, end=' ')
                    print(file=f)


if __name__ == '__main__':
    contr_exp_main('data')
