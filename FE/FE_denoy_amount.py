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


def new_deanonimize(num_rows=None):
    # load csv
    hist_df = pd.read_csv('./input/new_merchant_transactions.csv', nrows=num_rows)
    hist_df['purchase_amount'] = np.round(hist_df['purchase_amount'] / 0.00150265118 + 497.06,8)
    # fillna
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('D',inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['installments'].replace(-1, np.nan,inplace=True)
    hist_df['installments'].replace(999, np.nan,inplace=True)

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A':1, 'B':2, 'C':3,"D":0})

    
    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / (hist_df['installments']+1)


    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
    hist_df['month_diff'] += hist_df['month_lag']
    hist_df['nomonth_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
    # additional features
    hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
    hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']
    hist_df["duration_no"] =  hist_df['purchase_amount']*hist_df['nomonth_diff']
    hist_df['amount_month_ratio_no'] = hist_df['purchase_amount']/hist_df['nomonth_diff']
    hist_df['authorized_flag_amount'] = hist_df['authorized_flag'] * hist_df['purchase_amount']
    hist_df['category_3_amount'] = hist_df['category_3'] * hist_df['purchase_amount']
    hist_df['category_1_amount'] = hist_df['category_1'] * hist_df['purchase_amount']
    hist_df['category_2_amount'] = hist_df['category_2'] * hist_df['purchase_amount']
    hist_df['weekend_amount'] = hist_df['weekend'] * hist_df['purchase_amount']
    hist_df['monthlag_amount'] =   hist_df['purchase_amount']/(hist_df['month_lag']-1)
    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    aggs = {}

    aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
    aggs['price'] = ['sum','max','min','mean','var','skew']
    aggs['duration']=['sum','max','min','mean','var','skew']
    aggs['amount_month_ratio']=['sum','max','min','mean','var','skew']
    aggs["duration_no"] =['sum','max','min','mean','var','skew']
    aggs['amount_month_ratio_no'] =['sum','max','min','mean','var','skew']
    aggs['category_3'] = ['mean']
    aggs['authorized_flag_amount'] = ['sum','max','min','mean','var','skew']
    aggs['category_3_amount'] = ['sum','max','min','mean','var','skew']
    aggs['category_2_amount'] = ['sum','max','min','mean','var','skew']
    aggs['category_1_amount'] = ['sum','max','min','mean','var','skew']
    aggs['monthlag_amount'] = ['sum','max','min','mean','var','skew']
    for col in ['category_3']:
        hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col+'_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col+'_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col+'_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['new_amount'+ c for c in hist_df.columns]

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)


    return hist_df

def hist_deanonimize(start,end,num_rows=None):
    # load csv
    hist_df = pd.read_csv('./input/historical_transactions.csv', nrows=num_rows)
    hist_df['purchase_amount'] = np.round(hist_df['purchase_amount'] / 0.00150265118 + 497.06,8)
    # fillna
    hist_df['category_2'].fillna(1.0,inplace=True)
    hist_df['category_3'].fillna('D',inplace=True)
    hist_df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    hist_df['installments'].replace(-1, np.nan,inplace=True)
    hist_df['installments'].replace(999, np.nan,inplace=True)

    # Y/N to 1/0
    hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_1'] = hist_df['category_1'].map({'Y': 1, 'N': 0}).astype(int)
    hist_df['category_3'] = hist_df['category_3'].map({'A':1, 'B':2, 'C':3,"D":0})

    
    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])
    hist_df['month'] = hist_df['purchase_date'].dt.month
    hist_df['day'] = hist_df['purchase_date'].dt.day
    hist_df['hour'] = hist_df['purchase_date'].dt.hour
    hist_df['weekofyear'] = hist_df['purchase_date'].dt.weekofyear
    hist_df['weekday'] = hist_df['purchase_date'].dt.weekday
    hist_df['weekend'] = (hist_df['purchase_date'].dt.weekday >=5).astype(int)

    # additional features
    hist_df['price'] = hist_df['purchase_amount'] / (hist_df['installments']+1)


    hist_df['month_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
    hist_df['month_diff'] += hist_df['month_lag']
    hist_df['nomonth_diff'] = ((datetime.datetime.today() - hist_df['purchase_date']).dt.days)//30
    # additional features
    hist_df['duration'] = hist_df['purchase_amount']*hist_df['month_diff']
    hist_df['amount_month_ratio'] = hist_df['purchase_amount']/hist_df['month_diff']
    hist_df["duration_no"] =  hist_df['purchase_amount']*hist_df['nomonth_diff']
    hist_df['amount_month_ratio_no'] = hist_df['purchase_amount']/hist_df['nomonth_diff']
    hist_df['authorized_flag_amount'] = hist_df['authorized_flag'] * hist_df['purchase_amount']
    hist_df['category_3_amount'] = hist_df['category_3'] * hist_df['purchase_amount']
    hist_df['category_1_amount'] = hist_df['category_1'] * hist_df['purchase_amount']
    hist_df['category_2_amount'] = hist_df['category_2'] * hist_df['purchase_amount']
    hist_df['weekend_amount'] = hist_df['weekend'] * hist_df['purchase_amount']
    hist_df['monthlag_amount'] =   hist_df['purchase_amount']/(hist_df['month_lag']-1)
    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    aggs = {}

    aggs['purchase_amount'] = ['sum','max','min','mean','var','skew']
    aggs['price'] = ['sum','max','min','mean','var','skew']
    aggs['duration']=['sum','max','min','mean','var','skew']
    aggs['amount_month_ratio']=['sum','max','min','mean','var','skew']
    aggs["duration_no"] =['sum','max','min','mean','var','skew']
    aggs['amount_month_ratio_no'] =['sum','max','min','mean','var','skew']
    aggs['category_3'] = ['mean']
    aggs['authorized_flag_amount'] = ['sum','max','min','mean','var','skew']
    aggs['category_3_amount'] = ['sum','max','min','mean','var','skew']
    aggs['category_2_amount'] = ['sum','max','min','mean','var','skew']
    aggs['category_1_amount'] = ['sum','max','min','mean','var','skew']
    aggs['monthlag_amount'] = ['sum','max','min','mean','var','skew']
    for col in ['category_3']:
        hist_df[col+'_mean'] = hist_df.groupby([col])['purchase_amount'].transform('mean')
        hist_df[col+'_min'] = hist_df.groupby([col])['purchase_amount'].transform('min')
        hist_df[col+'_max'] = hist_df.groupby([col])['purchase_amount'].transform('max')
        hist_df[col+'_sum'] = hist_df.groupby([col])['purchase_amount'].transform('sum')
        aggs[col+'_mean'] = ['mean']

    hist_df = hist_df.reset_index()[hist_df["month_lag"]>=end][hist_df["month_lag"]<=start].groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_month'+str(start)+str(end)+ '_amount'+ c for c in hist_df.columns]

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    return hist_df