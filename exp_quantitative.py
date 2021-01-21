import os
import pickle
from problem import TSP
from dist import Map_Dist


def cat_exp_main(data_dir):
    m = 3
    metric = Map_Dist()

    fns_ = os.listdir(data_dir)
    fns = list(filter(lambda s: s.endswith('.pkl'), fns_))

    for fn in fns:
        ofn = fn[:-3] + 'csv'
        if ofn in fns_:
            continue
        with open(os.path.join(data_dir, fn), 'rb') as f:
            data = pickle.load(f)

        insts = data.keys()

        s = min(insts, key=len)
        print(s, end='')
        seed = TSP(data[s])
        with open(os.path.join(data_dir, ofn), 'w') as f:
            print('Inst_a,Inst_b,Distance,Baseline', file=f)
        for i in insts:
            print('*', end='')
            inst = TSP(data[i])
            if s == i:
                t = s + '_r0_#0'
            else:
                t = i
            for _ in range(m):
                res = (s, t, str(metric.dist(seed, inst)['dist']), str(abs(seed.hardness_est() - inst.hardness_est())))
                with open(os.path.join(data_dir, ofn), 'a') as f:
                    print(','.join(res), file=f)
        print()


if __name__ == '__main__':
    cat_exp_main('data')
