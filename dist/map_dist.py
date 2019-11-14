import numpy as np
from scipy.stats import pearsonr, ttest_ind
from scipy.linalg import norm
from matplotlib import pyplot as plt

from dist import Abstract_Dist
from solver import GA_Solver

from problem import Abstract_Prob, TSP


class Map_Dist(Abstract_Dist):
    # 1.0 for higher correlation 0.0 for better mean
    _alpha = 1.0
    _sample_size = 50
    _sample_iter = 50
    _pop_size = 300
    _max_iter = 50
    _radius = 0.1

    def __init__(self, sample_size=None, max_iter=None, radius=None):
        super().__init__()

        if sample_size:
            self._sample_size = sample_size
        if max_iter:
            self._max_iter = max_iter
        if radius:
            self._radius = radius

        print('sample size', self._sample_size)
        print('pi pop size', self._pop_size)
        print('iteration', self._max_iter)
        print('alpha', self._alpha)
        print('radius', self._radius)

    def dist(self, inst_a: Abstract_Prob, inst_b: Abstract_Prob):

        # sample from solution space
        splr = GA_Solver(pop_size=self._sample_size, max_iter=self._sample_iter, radius=self._radius)
        splr.solve(inst_a, niching='fs')
        spl_a = splr.samples
        splr.solve(inst_b, niching='fs')
        spl_b = splr.samples
        fit_a = np.array(list(map(lambda x: x.fitness.values[0], spl_a)))
        fit_b = np.array(list(map(lambda x: x.fitness.values[0], spl_b)))
        avg_a = np.mean(fit_a)
        avg_b = np.mean(fit_b)

        # print('fit_a mean', np.mean(fit_a))
        # print('fit_a std', np.std(fit_a))
        # print('fit_b mean', np.mean(fit_b))
        # print('fit_b std', np.std(fit_b))

        # optimise pi
        def two_term_form(pi):
            fit_trans = np.array([inst_b.eval(pi[s])[0] for s in spl_a])
            return [-(self._alpha * pearsonr(fit_a, fit_trans)[0] + (1 - self._alpha) * (avg_b / np.mean(fit_trans)))]

        def norm_form(pi):
            fit_trans = np.array([inst_b.eval(pi[s])[0] for s in spl_a])
            return [norm(fit_trans / avg_b - fit_a / avg_a) / self._sample_size]

        spl_slr = GA_Solver(pop_size=self._pop_size, max_iter=self._max_iter)
        pop = spl_slr.solve(Abstract_Prob(inst_b.n), eval=norm_form)
        fit = [ind.fitness.values[0] for ind in pop]
        # print(np.std(fit))
        idx = np.argmin(fit)

        return {'dist': fit[idx], 'pi': pop[idx]}


n = 50


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
