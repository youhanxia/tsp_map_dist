import numpy as np


def main(fn):
    res = []

    with open(fn, 'r') as f:
        l0 = f.readline().rstrip().split()
        print(l0[2:])
        for l in f:
            l = l.rstrip().split()
            b_res = float(l[1])
            res.append([float(r) / b_res for r in l[2:]])

    print('mean', np.mean(res, axis=0))
    print('std', np.std(res, axis=0))


if __name__ == '__main__':
    main('tune_spl_3.txt')
