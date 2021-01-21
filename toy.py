import numpy as np
import pickle
from scipy.stats import exponweib
from matplotlib import pyplot as plt

from problem import TSP


def weibull(u, shape, scale):
    return (shape / scale) * (u / scale)**(shape-1) * np.exp(-(u/scale)**shape)


def main(data_name, r=1):
    with open(data_name + '.pkl', 'rb') as f:
        data = pickle.load(f)

    insts = data.keys()
    seeds = list(filter(lambda x: '#' not in x, insts))

    for i in seeds:
        inst = TSP(data[i])
        dists = np.array(inst.dist_mat).flatten()

        # normalise to r == 1
        dists = dists / dists.max() * 2.0 * r

        _, shp, _, scl = exponweib.fit(dists, f0=1)
        print(i, np.abs(-0.509 * shp + 0.707))

        # plt.clf()
        # plt.title(i)
        # _ = plt.hist(dists, density=True)
        # x = np.linspace(dists.min(), dists.max(), 1000)
        # plt.plot(x, weibull(x, shp, scl))
        # plt.show()

    for i in range(5):
        inst = TSP(int(data_name.split('_')[-1]))
        dists = np.array(inst.dist_mat).flatten()
        dists = dists / dists.max() * 2.0 * r
        _, shp, _, scl = exponweib.fit(dists, f0=1)
        print('rand' + str(i), np.abs(-0.509 * shp + 0.707))


if __name__ == '__main__':
    data_name = 'exp_data_10'
    main(data_name)
    print()
    main(data_name, r=100)
