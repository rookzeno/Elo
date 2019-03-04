import datetime
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from utils import reduce_mem_usage,rmse,one_hot_encoder

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def hist_month(num_rows=None):
    # load csv
    hist_df = pd.read_csv('./input/historical_transactions.csv', nrows=num_rows)

    # fillna
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('A',inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['installments'].replace(-1, np.nan,inplace=True)
    hist_df['installments'].replace(999, np.nan,inplace=True)

    # trim
    hist_df['purchase_amount'] = hist_df['purchase_amount'].apply(lambda x: min(x, 0.8))

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A':0, 'B':1, 'C':2})

    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / (hist_df['installments']+1)
    aggs = {}
    for i in range(1,13,3):
        hist_df['2017' + str(i)]=(pd.to_datetime('2017-'+ str(i) + '-1')-hist_df['purchase_date']).dt.days.apply(lambda x: 1 if x > 0 and x < 91 else 0)
        hist_df['2017' + str(i) + "price"] = hist_df['2017' + str(i)]*hist_df['price']
        aggs['2017' + str(i) + "price"] = ['sum','mean','max' ,'min','var']
        aggs['2017' + str(i)] = ['sum','mean']
    for i in range(1,7,3):
        hist_df['2018' + str(i)]=(pd.to_datetime('2018-'+ str(i) + '-1')-hist_df['purchase_date']).dt.days.apply(lambda x: 1 if x > 0 and x < 91 else 0)
        hist_df['2018' + str(i) + "price"] = hist_df['2018' + str(i)]*hist_df['price']
        aggs['2018' + str(i) + "price"] = ['sum','mean','max' ,'min','var']
        aggs['2018' + str(i)] = ['sum','mean']

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)


    


    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_'+ c for c in hist_df.columns]

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    return hist_df