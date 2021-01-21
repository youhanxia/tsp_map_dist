import os
import pickle
import numpy as np
from problem import TSP
from dist import Map_Dist


def acc_exp_main(data_dir):
    m = 3
    metric = Map_Dist()
    ofn = 'dists.csv'

    with open(os.path.join(data_dir, ofn), 'w') as f:
        print('Inst_a,Inst_b,Distance,Baseline', file=f)

    fns_ = os.listdir(data_dir)
    fns = list(filter(lambda s: s.endswith('.pkl'), fns_))

    data = {}

    for fn in fns:
        with open(os.path.join(data_dir, fn), 'rb') as f:
            d = pickle.load(f)
        keys = list(d.keys())
        i = np.argmin([len(k) for k in keys])
        data[keys[i]] = d[keys[i]]
        # data.update()

    idx = list(data.keys())
    n = len(idx)
    met_dist_mat = np.zeros((n, n))
    har_dist_mat = np.zeros((n, n))

    print('#instances:', n)

    for i in range(n):
        for j in range(n):
            inst_i = TSP(data[idx[i]])
            inst_j = TSP(data[idx[j]])
            met_res = []
            har_res = []
            for k in range(m):
                met_res.append(metric.dist(inst_i, inst_j)['dist'])
                har_res.append(abs(inst_i.hardness_est() - inst_j.hardness_est()))

            met_dist_mat[i][j] = np.mean(met_res)
            har_dist_mat[i][j] = np.mean(har_res)
            s = (idx[i], idx[j], str(met_dist_mat[i][j]), str(har_dist_mat[i][j]))
            print(s)
            with open(os.path.join(data_dir, ofn), 'a') as f:
                print(','.join(s), file=f)


if __name__ == '__main__':
    acc_exp_main('data')
