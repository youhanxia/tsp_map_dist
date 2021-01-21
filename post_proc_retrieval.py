import os
import pickle
import numpy as np
import pandas as pd


def main(data_dir, data_fn):

    m = 25

    data = pd.read_csv(os.path.join(data_dir, data_fn))

    queries = data['Inst_a'].unique()

    stat = {
        'city': {1: [[], []], 5: [[], []], 10: [[], []]},
        'circuit': {1: [[], []], 5: [[], []], 10: [[], []]},
        'gaussian': {1: [[], []], 5: [[], []], 10: [[], []]},
        'uniform': {1: [[], []], 5: [[], []], 10: [[], []]},
    }

    for q in queries:
        t = q.split('_', 1)[0]
        d = data[data['Inst_a'] == q]
        if len(d) < 100:
            print('progress for next inst:', len(d)/100)
            continue

        d = d[d['Inst_b'] != q]

        # calc acc, recall,
        dist_ret = np.array(d['Distance'].argsort())
        bsl_ret = np.array(d['Baseline'].argsort())
        g_truth = np.array(d['Inst_b'].apply(lambda s: s.split('_', 1)[0] == t))

        # print('query:', q)
        for pos in [1, 5, 10]:
            prec = np.sum(g_truth[dist_ret][:pos]) / pos
            prec_b = np.sum(g_truth[bsl_ret][:pos]) / pos
            # print('perc at', pos, ':', prec)
            # print('base at', pos, ':', prec_b)
            stat[t][pos][0].append(prec)
            stat[t][pos][1].append(prec_b)

    for t in stat:
        print('for prob type', t)
        for pos in [1, 5, 10]:
            print('average precision at', pos, ':', np.mean(stat[t][pos][0]))
            print('std :', np.std(stat[t][pos][0]))
            print('average baseline precision at', pos, ':', np.mean(stat[t][pos][1]))
            print('std :', np.std(stat[t][pos][1]))
        print()


if __name__ == '__main__':
    main('data', 'dists.csv')
