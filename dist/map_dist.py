import numpy as np
from scipy.stats import pearsonr, ttest_ind
from scipy.linalg import norm
from matplotlib import pyplot as plt

from dist import Abstract_Dist, Geo_Dist
from solver import GA_Solver

from problem import Abstract_Prob, TSP


def vec_amp(v, factor=5):
    avg_v = np.mean(v)
    diff = v - avg_v
    # return diff * factor + avg_v
    return diff
    # return v


class Map_Dist(Abstract_Dist):
    lbl = 'mapping'

    # 1.0 for higher correlation 0.0 for better mean
    _alpha = 0.5
    _sample_size = 150
    _sample_iter = 250
    _pop_size = 50
    _max_iter = 50
    _radius = 0.1
    _verbose = False
    _obj = None

    def __init__(
            self,
            sample_size=None,
            sample_iter=None,
            pop_size=None,
            max_iter=None,
            radius=None,
            obj='two_term',
            verbose=False):
        super().__init__()

        if sample_size:
            self._sample_size = sample_size
        if sample_iter:
            self._sample_iter = sample_iter
        if pop_size:
            self._pop_size = pop_size
        if max_iter:
            self._max_iter = max_iter
        if radius:
            self._radius = radius
        self._obj = obj
        self._verbose = verbose

        if verbose:
            print('sample size', self._sample_size)
            print('pi pop size', self._pop_size)
            print('iteration', self._max_iter)
            print('alpha', self._alpha)
            print('radius', self._radius)

    def dist(self, inst_a: Abstract_Prob, inst_b: Abstract_Prob):

        init_pi = Geo_Dist().dist(inst_a, inst_b)['pi']

        # sample from solution space
        splr = GA_Solver(pop_size=self._sample_size, max_iter=self._sample_iter, radius=self._radius)
        spl_a = splr.solve(inst_a, niching='fs')['pop']
        # spl_a = splr.samples
        spl_b = splr.solve(inst_b, niching='fs')['pop']
        # spl_b = splr.samples
        # fit_a = np.array(list(map(lambda x: x.fitness.values[0], spl_a)))
        # fit_b = np.array(list(map(lambda x: x.fitness.values[0], spl_b)))
        fit_a = np.array([inst_a.eval(s)[0] for s in spl_a])
        fit_b = np.array([inst_b.eval(s)[0] for s in spl_b])
        avg_a = np.mean(fit_a)
        avg_b = np.mean(fit_b)

        # print('fit_a mean', np.mean(fit_a))
        # print('fit_a std', np.std(fit_a))
        # print('fit_b mean', np.mean(fit_b))
        # print('fit_b std', np.std(fit_b))

        # optimise pi
        def two_term_form(pi):
            fit_trans = np.array([inst_b.eval(pi[s])[0] for s in spl_a])
            return [-(self._alpha * pearsonr(vec_amp(fit_a), vec_amp(fit_trans))[0] + (1 - self._alpha) * (avg_b / np.mean(fit_trans)))]
            # return [-(self._alpha * pearsonr(fit_a, fit_trans)[0] + (1 - self._alpha) * (avg_b / np.mean(fit_trans)))]

        def norm_form(pi):
            fit_trans = np.array([inst_b.eval(pi[s])[0] for s in spl_a])
            return [norm(fit_trans / avg_b - fit_a / avg_a) / self._sample_size]

        # print('two term form', two_term_form(init_pi))
        # print('norm form', norm_form(init_pi))

        if self._obj == 'norm':
            obj_func = norm_form
        else:
            obj_func = two_term_form

        spl_slr = GA_Solver(pop_size=self._pop_size, max_iter=self._max_iter)
        res = spl_slr.solve(Abstract_Prob(inst_b.n), eval=obj_func, seed=init_pi)

        return {'dist': res['fitness'], 'pi': res['tour']}


n = 10


def main0():
    ia = TSP(n)
    ib = TSP(n)
    metric = Map_Dist()
    res_ident = metric.dist(ia, ia)
    res_indep = metric.dist(ia, ib)

    print('ident:', res_ident['dist'], res_ident['pi'])
    print('indep:', res_indep['dist'], res_indep['pi'])


def main1():
    ident = []
    indep = []

    print('prob size', n)

    metric = Map_Dist()

    for i in range(30):
        print('\r', i, end='')
        ia = TSP(n)
        ib = TSP(n)
        id_res = metric.dist(ia, ia)
        in_res = metric.dist(ia, ib)
        ident.append(id_res['dist'])
        indep.append(in_res['dist'])

    print('\r', end='')
    print('identical set mean', np.mean(ident))
    print('identical set std', np.std(ident))
    print('independent set mean', np.mean(indep))
    print('independent set std', np.std(indep))
    print('p =', ttest_ind(ident, indep)[1])

    # plt.boxplot([ident, indep], labels=['identical', 'independent'])
    # plt.show()


if __name__ == '__main__':
    main0()
    # main1()
