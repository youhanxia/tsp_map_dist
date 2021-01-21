import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from problem import TSP
from dist import Map_Dist
from solver import *


def exp_main(data_name, out_name):
    with open(data_name + '.pkl', 'rb') as f:
        data = pickle.load(f)

    insts = data.keys()

    seeds = list(filter(lambda x: '#' not in x, insts))

    metric = Map_Dist()
    solver = GA_Solver()
    heuristics = [NN_Heuristic(), Gr_Heuristic(), NI_Heuristic(), FI_Heuristic(), DMST_Heuristic()]

    res = dict()

    m = 5

    for ia in seeds:
        res_ia = dict()
        print(ia)
        inst_a = TSP(data[ia])
        print('other inst init')
        for ib in seeds:
            if ia == ib:
                continue
            inst_b = TSP(data[ib])
            pi = metric.dist(inst_b, inst_a)['pi']
            init = pi[inst_b.optimal_tour]
            iter_avg = []
            for k in range(m):
                sol = solver.solve(inst_a, seed=init)
                iter_avg.append(sol['iter_avg'])
            iter_avg = np.array(iter_avg)
            res_ia[ib] = (init, iter_avg.sum(axis=0))
        print('closest inst init')
        for ib in insts:
            if ia not in ib or ('r1' not in ib and 't1' not in ib):
                continue
            inst_b = TSP(data[ib])
            pi = metric.dist(inst_b, inst_a)['pi']
            init = pi[inst_b.optimal_tour]
            iter_avg = []
            for k in range(m):
                sol = solver.solve(inst_a, seed=init)
                iter_avg.append(sol['iter_avg'])
            iter_avg = np.array(iter_avg)
            res_ia[ib] = (init, iter_avg.sum(axis=0))
        print('heuristic init')
        for heuristic in heuristics:
            init = heuristic.solve(inst_a)['tour']
            iter_avg = []
            for k in range(m):
                sol = solver.solve(inst_a, seed=init)
                iter_avg.append(sol['iter_avg'])
            iter_avg = np.array(iter_avg)
            res_ia[heuristic.__class__.__name__] = (init, iter_avg.sum(axis=0))
        print('random init')
        iter_avg = []
        for k in range(m):
            sol = solver.solve(inst_a)
            iter_avg.append(sol['iter_avg'])
        iter_avg = np.array(iter_avg)
        res_ia['random'] = (init, iter_avg.sum(axis=0))
        res[ia] = res_ia

    with open(out_name, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


def post_proc(out_name):
    with open(out_name, 'rb') as f:
        data = pickle.load(f)

    for ia in data.keys():
        plt.clf()
        plt.figure(figsize=(15, 15))
        plt.title(ia)
        plt.xlabel('iteration')
        plt.ylabel('average fitness')
        for ib in data[ia].keys():
            y = data[ia][ib][1]
            if ia in ib:
                ltype = '-'
            elif 'Heuristic' in ib:
                ltype = '.'
            elif 'random' in ib:
                ltype = '+'
            else:
                ltype = '.-'
            plt.plot(y, ltype, label=ib)
        plt.legend()
        plt.savefig(os.path.join('plots', ia + '_init.png'))


if __name__ == '__main__':
    # exp_main('exp_data_50', 'res_init_50.pkl')
    post_proc('res_init_50.pkl')
