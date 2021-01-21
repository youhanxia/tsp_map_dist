import os
import re
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def post_proc(data_dir):
    fns_ = os.listdir(data_dir)
    fns = list(filter(lambda s: s.endswith('.csv'), fns_))

    for fn in fns:
        data = pd.read_csv(os.path.join(data_dir, fn))

        data['Ctrl_type'] = data['Inst_b'].apply(lambda x: x.split('_')[-2][0])
        data['Ctrl_param'] = data['Inst_b'].apply(lambda x: int(x.split('_')[-2][1]))

        plt.clf()
        plt.title(fn)
        for ctrl in ['r', 't']:
            dc = data[data['Ctrl_type'] == ctrl]
            print(dc.dtypes)
            dc = dc[['Ctrl_param', 'Distance', ' Baseline']]
            print(dc)
            dc = dc.groupby('Ctrl_param').mean()
            plt.plot(dc.index, dc['Distance'] / np.max(dc['Distance']), '-', label='Map_Dist_' + ctrl)
            plt.plot(dc.index, dc[' Baseline'] / np.max(dc[' Baseline']), '.-', label='Baseline_' + ctrl)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    post_proc('data')
