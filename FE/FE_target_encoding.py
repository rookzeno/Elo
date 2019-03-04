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
from tqdm import tqdm_notebook as tqdm
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import feather
from utils import reduce_mem_usage,rmse

def target_encoding_hist(num_rows=None):
    df = feather.read_dataframe("./feature/traintest.feather")
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    train_df['rounded_target'] = train_df['target'].round(0)
    train_df = train_df.sort_values('rounded_target').reset_index(drop=True)
    vc = train_df['rounded_target'].value_counts()
    vc = dict(sorted(vc.items()))
    df = pd.DataFrame()
    train_df['indexcol'],i = 0,1
    for k,v in vc.items():
        step = train_df.shape[0]/v
        indent = train_df.shape[0]/(v+1)
        df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=10).reset_index(drop=True)
        for j in range(0, v):
            df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
        df = pd.concat([df2,df])
        i+=1
    train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
    del train_df['indexcol'], train_df['rounded_target']

    folds = KFold(n_splits= 7, shuffle=False, random_state=326)
    sub_preds = np.zeros(test_df.shape[0])
    oof_preds = np.zeros(train_df.shape[0])
    moto = pd.DataFrame()
    prior = train_df["target"].mean()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['outliers'])):
        hist_df = pd.read_csv('./input/historical_transactions.csv', nrows=num_rows)
        train_x, train_y = train_df.iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_df['target'].iloc[valid_idx]
        card = train_x[["card_id","target"]].values
        cata = {}
        for i,j in card:
            cata[i] = j
        hist_df["target"] = hist_df["card_id"].apply(lambda x : cata[x] if x in cata else np.nan)
        histtar = [#'authorized_flag',
                'city_id', 'category_1', 'installments','authorized_flag',
            'category_3', 'merchant_category_id', 'month_lag',"merchant_id",
                'category_2', 'state_id',
            'subsector_id']
        prior = hist_df["target"].mean()
        aggs = {}
        sou = []
        for i in histtar:
            averages = hist_df.groupby([i])['target'].agg(["mean", "count"])
            smoothing = 1 / (1 + np.exp(-(averages["count"] - 85) / 1))
            averages[i] = prior * (1 - smoothing) + averages["mean"] * smoothing
            averages.drop(["mean", "count"], axis=1, inplace=True)
            averages = averages.to_dict()
            averages = averages[i]
            hist_df[i+"taren"] = hist_df[i].map(averages)
            aggs[i+"taren"] = ['sum','mean','max' ,'min','var']
            sou.append(i+"taren")
        for i in range(len(sou)):
            for j in range(len(sou)):
                if i >= j:
                    continue
                ii = sou[i]
                jj = sou[j]
                hist_df[ii+jj+"sou"] = hist_df[ii]/hist_df[jj]
                aggs[ii+jj+"sou"] = ['sum','mean','max' ,'min','var']
        hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)
        hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
        hist_df.columns = ['hist_target_encoding_'+ c for c in hist_df.columns]

        # reduce memory usage
        hist_df = reduce_mem_usage(hist_df)
        valid_x = pd.merge(valid_x,hist_df,on="card_id",how='left')
        moto = moto.append(valid_x)

    prior = train_df["target"].mean()
    card = train_df[["card_id","target"]].values
    hist_df = pd.read_csv('./input/historical_transactions.csv', nrows=num_rows)
    cata = {}
    for i,j in card:
        cata[i] = j
    hist_df["target"] = hist_df["card_id"].apply(lambda x : cata[x] if x in cata else np.nan)
    histtar = [#'authorized_flag', 
            'city_id', 'category_1', 'installments','authorized_flag',
        'category_3', 'merchant_category_id', 'month_lag',"merchant_id",
            'category_2', 'state_id',
        'subsector_id']
    prior = hist_df["target"].mean()
    aggs = {}
    sou = []
    for i in histtar:
        averages = hist_df.groupby([i])['target'].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(averages["count"] - 85) / 1))
        averages[i] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        averages = averages.to_dict()
        averages = averages[i]
        hist_df[i+"taren"] = hist_df[i].map(averages)
        aggs[i+"taren"] = ['sum','mean','max' ,'min','var']
        sou.append(i+"taren")
    for i in range(len(sou)):
        for j in range(len(sou)):
            if i >= j:
                continue
            ii = sou[i]
            jj = sou[j]
            hist_df[ii+jj+"sou"] = hist_df[ii]/hist_df[jj]
            aggs[ii+jj+"sou"] = ['sum','mean','max' ,'min','var']
    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['hist_target_encoding_'+ c for c in hist_df.columns]

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)
    test_df = pd.merge(test_df,hist_df,on="card_id",how='left')
    moto = moto.append(test_df)

    kore = moto.columns[25:]
    syutu = moto[["card_id"] + list(kore)]
    df = syutu.reset_index(drop=True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type =='float16':
            df[col] = df[col].astype(np.float32)
    return df
#    df.to_feather("./feature/hist_oof_taren.feather")

def target_encoding_new(num_rows=None):
    df = feather.read_dataframe("./feature/traintest.feather")
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    train_df['rounded_target'] = train_df['target'].round(0)
    train_df = train_df.sort_values('rounded_target').reset_index(drop=True)
    vc = train_df['rounded_target'].value_counts()
    vc = dict(sorted(vc.items()))
    df = pd.DataFrame()
    train_df['indexcol'],i = 0,1
    for k,v in vc.items():
        step = train_df.shape[0]/v
        indent = train_df.shape[0]/(v+1)
        df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=10).reset_index(drop=True)
        for j in range(0, v):
            df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
        df = pd.concat([df2,df])
        i+=1
    train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
    del train_df['indexcol'], train_df['rounded_target']

    folds = KFold(n_splits= 7, shuffle=False, random_state=326)
    sub_preds = np.zeros(test_df.shape[0])
    oof_preds = np.zeros(train_df.shape[0])
    moto = pd.DataFrame()
    prior = train_df["target"].mean()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['outliers'])):
        hist_df = pd.read_csv('./input/new_merchant_transactions.csv', nrows=num_rows)
        train_x, train_y = train_df.iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_df['target'].iloc[valid_idx]
        card = train_x[["card_id","target"]].values
        cata = {}
        for i,j in card:
            cata[i] = j
        hist_df["target"] = hist_df["card_id"].apply(lambda x : cata[x] if x in cata else np.nan)
        histtar = [#'authorized_flag',
                'city_id', 'category_1', 'installments','authorized_flag',
            'category_3', 'merchant_category_id', 'month_lag',"merchant_id",
                'category_2', 'state_id',
            'subsector_id']
        prior = hist_df["target"].mean()
        aggs = {}
        sou = []
        for i in histtar:
            averages = hist_df.groupby([i])['target'].agg(["mean", "count"])
            smoothing = 1 / (1 + np.exp(-(averages["count"] - 85) / 1))
            averages[i] = prior * (1 - smoothing) + averages["mean"] * smoothing
            averages.drop(["mean", "count"], axis=1, inplace=True)
            averages = averages.to_dict()
            averages = averages[i]
            hist_df[i+"taren"] = hist_df[i].map(averages)
            aggs[i+"taren"] = ['sum','mean','max' ,'min','var']
            sou.append(i+"taren")
        for i in range(len(sou)):
            for j in range(len(sou)):
                if i >= j:
                    continue
                ii = sou[i]
                jj = sou[j]
                hist_df[ii+jj+"sou"] = hist_df[ii]/hist_df[jj]
                aggs[ii+jj+"sou"] = ['sum','mean','max' ,'min','var']
        hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)
        hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
        hist_df.columns = ['new_target_encoding_'+ c for c in hist_df.columns]

        # reduce memory usage
        hist_df = reduce_mem_usage(hist_df)
        valid_x = pd.merge(valid_x,hist_df,on="card_id",how='left')
        moto = moto.append(valid_x)

    prior = train_df["target"].mean()
    card = train_df[["card_id","target"]].values
    hist_df = pd.read_csv('./input/new_merchant_transactions.csv', nrows=num_rows)
    cata = {}
    for i,j in card:
        cata[i] = j
    hist_df["target"] = hist_df["card_id"].apply(lambda x : cata[x] if x in cata else np.nan)
    histtar = [#'authorized_flag', 
            'city_id', 'category_1', 'installments','authorized_flag',
        'category_3', 'merchant_category_id', 'month_lag',"merchant_id",
            'category_2', 'state_id',
        'subsector_id']
    prior = hist_df["target"].mean()
    aggs = {}
    sou = []
    for i in histtar:
        averages = hist_df.groupby([i])['target'].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(averages["count"] - 85) / 1))
        averages[i] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        averages = averages.to_dict()
        averages = averages[i]
        hist_df[i+"taren"] = hist_df[i].map(averages)
        aggs[i+"taren"] = ['sum','mean','max' ,'min','var']
        sou.append(i+"taren")
    for i in range(len(sou)):
        for j in range(len(sou)):
            if i >= j:
                continue
            ii = sou[i]
            jj = sou[j]
            hist_df[ii+jj+"sou"] = hist_df[ii]/hist_df[jj]
            aggs[ii+jj+"sou"] = ['sum','mean','max' ,'min','var']
    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['new_target_encoding_'+ c for c in hist_df.columns]

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)
    test_df = pd.merge(test_df,hist_df,on="card_id",how='left')
    moto = moto.append(test_df)

    kore = moto.columns[25:]
    syutu = moto[["card_id"] + list(kore)]
    df = syutu.reset_index(drop=True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type =='float16':
            df[col] = df[col].astype(np.float32)
    return df
#    df.to_feather("./feature/hist_oof_taren.feather")