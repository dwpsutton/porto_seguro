# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import gc

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

from time import time
from random import choice
from scipy.stats import randint as sp_randint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def run_pipe3(df_tn_z,df_tt_z):

    np.random.seed(17)

    # replace -1 for NaN

    # train set
    df_tn_z.replace(-1, np.NaN, inplace = True)

    # test set
    df_tt_z.replace(-1, np.NaN, inplace = True)
    if "target" in df_tt_z.columns:
        Y_tt= df_tt_z.target.values
    else:
        Y_tt= None

    # -1 can be changed to 0 for features where there is no category "0",
    # and features that have numerical values. Scrip below identifies such
    # features as well as those where -1 shouldn't be changed.

    # list with features
    zero_list = ['ps_ind_02_cat', 'ps_reg_03', 'ps_car_12', 'ps_car_12', 'ps_car_14',] # -1 can be changed for 0 in this features

    minus_one = ['ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat',
                 'ps_car_03_cat', 'ps_car_07_cat', 'ps_car_05_cat', 'ps_car_09_cat',
                 'ps_car_11'] # these features already have 0 as value, thus -1 shouldn't be changed


    # fill in missing values with 0 or -1

    # train set
    df_tn_z[minus_one] = df_tn_z[minus_one].fillna(-1)
    df_tn_z[zero_list] = df_tn_z[zero_list].fillna(0)

    # test set
    df_tt_z[minus_one] = df_tt_z[minus_one].fillna(-1)
    df_tt_z[zero_list] = df_tt_z[zero_list].fillna(0)


    # group features by nature
    cat_f = ['ps_ind_02_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat',
             'ps_car_03_cat',  'ps_car_04_cat','ps_car_05_cat', 'ps_car_06_cat',
             'ps_car_07_cat', 'ps_car_08_cat','ps_car_09_cat', 'ps_car_10_cat',
             'ps_car_11_cat']
    bin_f = ['ps_ind_04_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
             'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
             'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',
             'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',
             'ps_calc_19_bin', 'ps_calc_20_bin']
    ord_f = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_car_11']

    cont_f = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13',
              'ps_car_14', 'ps_car_15',  'ps_calc_01', 'ps_calc_02', 'ps_calc_03',
              'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08',
              'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',
              'ps_calc_14']

    # transform categorical values to dummies
    df_tn_proc = df_tn_z.copy().drop(['id', 'target'], axis = 1)
    df_tt_proc = df_tt_z.copy().drop(['id'], axis = 1)
    df_all_proc = pd.concat((df_tn_proc, df_tt_proc), axis=0, ignore_index=True)

    for i in cat_f:
        d = pd.get_dummies(df_all_proc[i], prefix = i, prefix_sep='_')
        df_all_proc.drop(i, axis = 1, inplace = True)
        df_all_proc = df_all_proc.merge(d, right_index=True, left_index=True)

    # prepare X and Y
    X = df_all_proc[:df_tn_z.shape[0]].copy()
    X_tt = df_all_proc[df_tn_z.shape[0]:].copy()
    Y = df_tn_z['target'].copy()

    return X,Y,X_tt,Y_tt

# formula for Gini Coefficient (https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703)
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def test():
    gc.enable()
    
    trn_df = pd.read_csv("../data/train.csv")
    sub_df = pd.read_csv("../data/test.csv")
    return run_pipe3(trn_df,sub_df)

if __name__ == '__main__':
    test()



