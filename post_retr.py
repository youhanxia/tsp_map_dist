import numpy as np


def main(fn):
    res = dict()

    with open(fn, 'r') as f:
        print('query metric min_dist min_dist_fit min_dist_prob min_fit min_fit_prob')
        l0 = f.readline().rstrip().split()
        res = dict()
        cur_query = None
        cur_metric = None
        min_dist = np.inf
        min_dist_fit = np.inf
        min_dist_prob = None
        min_fit = np.inf
        min_fit_prob = None
        for l in f:
            l = l.rstrip().split()
            dist = float(l[3]) if l[3] != 'None' else 0.0
            fit = float(l[4])
            if cur_query != l[0] or cur_metric != l[1]:
                if cur_query is not None:
                    if cur_query not in res:
                        res[cur_query] = dict()
                    res[cur_query][cur_metric] = {
                        'min_dist': min_dist,
                        'min_dist_fit': min_dist_fit,
                        'min_fit': min_fit,
                    }
                    print(cur_query, cur_metric, min_dist, min_dist_fit, min_dist_prob, min_fit, min_fit_prob)
                cur_query = l[0]
                cur_metric = l[1]
                min_dist = np.inf
                min_dist_fit = np.inf
                min_dist_prob = None
                min_fit = np.inf
                min_fit_prob = None
            if min_dist > dist:
                min_dist = dist
                min_dist_fit = fit
                min_dist_prob = l[2]
            if min_fit > fit:
                min_fit = fit
                min_fit_prob = l[2]

    mets = res['gaussian_11'].keys()

    with open(fn[:-4] + '_tab.txt', 'w') as f:
        print('query', file=f, end=' ')
        for m in mets:
            print(m, file=f, end=' ')
        print(file=f)
        for q in res:
            print(q, file=f, end=' ')
            for m in mets:
                print(res[q][m]['min_fit'], file=f, end=' ')
            print(file=f)


if __name__ == '__main__':
    main('retr_0.txt')
