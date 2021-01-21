import numpy as np
import pandas as pd


def formatting(fn):
    with open(fn, 'r') as fin:
        with open(fn[:-4] + '_1.txt', 'w') as fout:
            l0 = fin.readline().rstrip()
            print(l0, file=fout)
            i = 2
            for l in fin:
                l_s = l.rstrip().split()
                l_s.pop(i)
                if i == 1:
                    del l_s[3: 42]
                i = 3 - i
                print(' '.join(l_s), file=fout)


def main(fn):
    with open(fn, 'r') as f:
        l0 = f.readline().rstrip().split()
        names = []
        res = []
        for l in f:
            l = l.rstrip().split()
            names.append(l[:3])
            scores = list(map(float, l[3:]))
            res.append([scores[i] < scores[i + 1] and scores[i] < scores[i + 2] for i in range(0, len(scores), 3)])
    res = np.array(res, dtype=float)
    # res = np.array(res)
    # print(res)

    df = pd.DataFrame(names, columns=l0[:3])
    df[l0[3:]] = pd.DataFrame(res)
    df['cat_a'] = df['inst_a'].apply(lambda x: x.split('_', 1)[0])
    df['cat_c'] = df['inst_c'].apply(lambda x: x.split('_', 1)[0])
    df['same_orig'] = df.apply(lambda row: row['inst_a'].split('_', 2)[1] == row['inst_b'].split('_', 2)[1], axis=1)

    df = df.drop(columns=['inst_a', 'inst_b', 'inst_c'])
    # print(df.iloc[0])

    df_res = pd.DataFrame(df.groupby(['cat_a', 'cat_c', 'same_orig']).mean())
    df_res.to_csv(fn[:-4] + '_stats.csv')
    df_count = pd.DataFrame(df[['cat_a', 'cat_c', 'same_orig', 'mapping']].groupby(['cat_a', 'cat_c', 'same_orig']).count())
    df_count.to_csv(fn[:-4] + '_count.csv')


if __name__ == '__main__':
    main('contr_0.txt')
    # formatting('contr_0.txt')
