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
from utils import reduce_mem_usage,rmse

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']

def make_new_mer(num_rows=None):
    # load csv
    new_df = pd.read_csv('./input/new_merchant_transactions.csv', nrows=num_rows)

    # fillna
    new_df['category_2'].fillna(1.0,inplace=True)
    new_df['category_3'].fillna('A',inplace=True)
    new_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    new_df['installments'].replace(-1, np.nan,inplace=True)
    new_df['installments'].replace(999, np.nan,inplace=True)

    # trim
    new_df['purchase_amount'] = np.round(new_df['purchase_amount'] / 0.00150265118 + 497.06,8)

    # Y/N to 1/0
    new_df['authorized_flag'] = new_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    new_df['category_1'] = new_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    new_df['category_3'] = new_df['category_3'].map({'A':0, 'B':1, 'C':2})

    # datetime features
    new_df['purchase_date'] = pd.to_datetime(new_df['purchase_date'])
    new_df['month'] = new_df['purchase_date'].dt.month
    new_df['day'] = new_df['purchase_date'].dt.day
    new_df['hour'] = new_df['purchase_date'].dt.hour
    new_df['weekofyear'] = new_df['purchase_date'].dt.weekofyear
    new_df['weekday'] = new_df['purchase_date'].dt.weekday
    new_df['weekend'] = (new_df['purchase_date'].dt.weekday >=5).astype(int)

    # additional features
    new_df['price'] = new_df['purchase_amount'] / (new_df['installments']+1)

    #Christmas : December 25 2017
    new_df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-new_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Mothers Day: May 14 2017
    new_df['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-new_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #fathers day: August 13 2017
    new_df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-new_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Childrens day: October 12 2017
    new_df['Children_day_2017']=(pd.to_datetime('2017-10-12')-new_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Valentine's Day : 12th June, 2017
    new_df['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-new_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Black Friday : 24th November 2017
    new_df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - new_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    #2018
    #Mothers Day: May 13 2018
    new_df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-new_df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    new_df['month_diff'] = ((datetime.datetime.today() - new_df['purchase_date']).dt.days)//30
    new_df['month_diff'] += new_df['month_lag']

    # additional features
    new_df['duration'] = new_df['purchase_amount']*new_df['month_diff']
    new_df['amount_month_ratio'] = new_df['purchase_amount']/new_df['month_diff']

    # reduce memory usage
    new_df = reduce_mem_usage(new_df)

    col_unique =['subsector_id', 'merchant_category_id']
    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}
    for col in col_unique:
        aggs[col] = ['nunique']

    for col in col_seas:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
    aggs['installments'] = ['sum','max','mean','var','skew']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var','skew']
    aggs['month_diff'] = ['max','min','mean','var','skew']
    aggs['authorized_flag'] = ['mean']
    aggs['weekend'] = ['mean'] # overwrite
    aggs['weekday'] = ['mean'] # overwrite
    aggs['day'] = ['nunique', 'mean', 'min'] # overwrite
    aggs['category_1'] = ['mean']
    aggs['category_2'] = ['mean']
    aggs['category_3'] = ['mean']
    aggs['card_id'] = ['nunique']
    aggs['merchant_id'] = ["count","size"]
    aggs['price'] = ['sum','mean','max','min','var']
    aggs['Christmas_Day_2017'] = ['mean']
    aggs['Mothers_Day_2017'] = ['mean']
    aggs['fathers_day_2017'] = ['mean']
    aggs['Children_day_2017'] = ['mean']
    aggs['Valentine_Day_2017'] = ['mean']
    aggs['Black_Friday_2017'] = ['mean']
    aggs['Mothers_Day_2018'] = ['mean']
    aggs['duration']=['mean','min','max','var','skew']
    aggs['amount_month_ratio']=['mean','min','max','var','skew']

    for col in ['category_2','category_3']:
        new_df[col+'_mean'] = new_df.groupby([col])['purchase_amount'].transform('mean')
        new_df[col+'_min'] = new_df.groupby([col])['purchase_amount'].transform('min')
        new_df[col+'_max'] = new_df.groupby([col])['purchase_amount'].transform('max')
        new_df[col+'_sum'] = new_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    new_df = new_df.reset_index().groupby('merchant_id').agg(aggs)

    # change column name
    new_df.columns = pd.Index([e[0] + "_" + e[1] for e in new_df.columns.tolist()])
    new_df.columns = ['new_'+ c for c in new_df.columns]

    new_df['new_purchase_date_diff'] = (new_df['new_purchase_date_max']-new_df['new_purchase_date_min']).dt.days
    new_df['new_purchase_date_average'] = new_df['new_purchase_date_diff']/new_df['new_merchant_id_size']
    new_df['new_purchase_date_uptonow'] = (datetime.datetime.today()-new_df['new_purchase_date_max']).dt.days
    new_df['new_purchase_date_uptomin'] = (datetime.datetime.today()-new_df['new_purchase_date_min']).dt.days

    # reduce memory usage
    new_df = reduce_mem_usage(new_df)
    new_df.reset_index().to_feather("./input/new_merchant.feather")
    return 0

