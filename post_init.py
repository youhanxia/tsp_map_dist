import pandas as pd


def main(fn):
    df = pd.read_csv(fn, sep=' ')
    df = df.drop(columns=df.keys()[-1])
    keys = df.keys()
    # print(keys)

    for k in keys[3:]:
        df[k] /= df['best']

    df = df.drop(columns='best')

    df['mod_lvl'] = df.apply(lambda row: 0 if row['inst_a'] == row['inst_b'] else int(row['inst_b'][-4]), axis=1)
    df['cat'] = df.apply(lambda row: row['inst_a'].split('_')[0], axis=1)

    # print(df.iloc[1])

    df_res = pd.DataFrame(df.groupby(['mod_lvl', 'cat']).mean())
    df_res.to_csv(fn[:-4] + '_stats.csv')


if __name__ == '__main__':
    main('init_0.txt')
