import numpy as np
import pandas as pd
import feather

import datetime
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
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

def date_targetencoding(num_rows=None):
    pd.set_option('display.float_format', '{:.10f}'.format)
    train = feather.read_dataframe("./feature/traintest.feather")
    historical_transactions = pd.read_csv('./input/historical_transactions.csv', nrows=num_rows)
    new_merchant_transactions = pd.read_csv('./input/new_merchant_transactions.csv', nrows=num_rows)

    # fast way to get last historic transaction / first new transaction
    last_hist_transaction = historical_transactions.groupby('card_id').agg({'month_lag' : 'max', 'purchase_date' : 'max'}).reset_index()
    last_hist_transaction.columns = ['card_id', 'hist_month_lag', 'hist_purchase_date']
    first_new_transaction = new_merchant_transactions.groupby('card_id').agg({'month_lag' : 'min', 'purchase_date' : 'min'}).reset_index()
    first_new_transaction.columns = ['card_id', 'new_month_lag', 'new_purchase_date']

    # converting to datetime
    last_hist_transaction['hist_purchase_date'] = pd.to_datetime(last_hist_transaction['hist_purchase_date']) 
    first_new_transaction['new_purchase_date'] = pd.to_datetime(first_new_transaction['new_purchase_date']) 

    # substracting month_lag for each row
    last_hist_transaction['observation_date'] = \
        last_hist_transaction.apply(lambda x: x['hist_purchase_date']  - pd.DateOffset(months=x['hist_month_lag']), axis=1)

    first_new_transaction['observation_date'] = \
        first_new_transaction.apply(lambda x: x['new_purchase_date']  - pd.DateOffset(months=x['new_month_lag']-1), axis=1)

    last_hist_transaction['observation_date'] = last_hist_transaction['observation_date'].dt.to_period('M').dt.to_timestamp() + pd.DateOffset(months=1)
    first_new_transaction['observation_date'] = first_new_transaction['observation_date'].dt.to_period('M').dt.to_timestamp()

    train = train.merge(last_hist_transaction, on = 'card_id',how="outer")

    df = train

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
        df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=1199).reset_index(drop=True)
        for j in range(0, v):
            df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
        df = pd.concat([df2,df])
        i+=1
    train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
    del train_df['indexcol'], train_df['rounded_target']
    #target_encoding
    folds = KFold(n_splits= 7, shuffle=False, random_state=326)
    sub_preds = np.zeros(test_df.shape[0])
    oof_preds = np.zeros(train_df.shape[0])
    moto = pd.DataFrame()
    prior = train_df["target"].mean()
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df['outliers'])):
        train_x, train_y = train_df.iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], train_df['target'].iloc[valid_idx]
        prior = train_x["target"].mean()
        averages = train_x.groupby(["observation_date"])['target'].agg(["mean", "count"])
        smoothing = 1 / (1 + np.exp(-(averages["count"] - 85) / 1))
        averages[i] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        averages = averages.to_dict()
        averages = averages[i]
        valid_x["observation_date_target"] = valid_x["observation_date"].map(averages)
        moto = moto.append(valid_x)

    prior = train_df["target"].mean()
    averages = train_df.groupby(["observation_date"])['target'].agg(["mean", "count"])
    smoothing = 1 / (1 + np.exp(-(averages["count"] - 85) / 1))
    averages[i] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    averages = averages.to_dict()
    averages = averages[i]
    test_df["observation_date_target"] = test_df["observation_date"].map(averages)
    moto = moto.append(test_df)
    return moto[["card_id","observation_date_target"]]