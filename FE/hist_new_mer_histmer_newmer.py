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
import feather
from tqdm import tqdm_notebook as tqdm
from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from utils import reduce_mem_usage,rmse,one_hot_encoder
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def hist_new_hist_mer(num_rows=None):
    hist_df = pd.read_csv('./input/new_merchant_transactions.csv', nrows=num_rows)

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
    merchant = feather.read_dataframe("./input/newmer+hist_merchant.feather")
    merchant.set_index("merchant_id",inplace=True)
#    merchant.fillna(0,inplace=True)
    merdict = {}
    value = merchant.values
    indexa = merchant.index
    for i in tqdm(range(len(indexa))):
        merdict[indexa[i]] = value[i]
    tuika = []
    sono = hist_df["merchant_id"].values
    for i in tqdm(range(len(hist_df))):
        try:
            tuika.append(merdict[sono[i]])
        except:
            tuika.append([[np.nan]*len(merchant.columns)][0])
    tuika = np.array(tuika)
    mer = []
    aggs = {}
    for j in range(len(merchant.columns)):
        i = merchant.columns[j]
        if i.find("date_m")!=-1:
            continue
            hist_df["mer1"+i] = tuika[:,j]
            aggs["mer1"+i] = ["min","max"]
            hist_df["mer1"+i] = pd.to_datetime(hist_df["mer1"+i])
        else:
            hist_df["mer1"+i] = tuika[:,j]
            hist_df["mer1"+i] = hist_df["mer1"+i].astype(np.float32)
        aggs["mer1"+i] = ["sum","mean","min","max"]
    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])

    
    hist_df = reduce_mem_usage(hist_df)


    
    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['new_'+ c for c in hist_df.columns]

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    return hist_df


def hist_new_new_mer(num_rows=None):
    hist_df = pd.read_csv('./input/new_merchant_transactions.csv', nrows=num_rows)

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
    merchant = feather.read_dataframe("./input/newmer+hist_merchant.feather")
    merchant.set_index("merchant_id",inplace=True)
#    merchant.fillna(0,inplace=True)
    merdict = {}
    value = merchant.values
    indexa = merchant.index
    for i in tqdm(range(len(indexa))):
        merdict[indexa[i]] = value[i]
    tuika = []
    sono = hist_df["merchant_id"].values
    for i in tqdm(range(len(hist_df))):
        try:
            tuika.append(merdict[sono[i]])
        except:
            tuika.append([[np.nan]*len(merchant.columns)][0])
    tuika = np.array(tuika)
    mer = []
    aggs = {}
    for j in range(len(merchant.columns)):
        i = merchant.columns[j]
        if i.find("date_m")!=-1:
            continue
            hist_df["mer1"+i] = tuika[:,j]
            aggs["mer1"+i] = ["min","max"]
            hist_df["mer1"+i] = pd.to_datetime(hist_df["mer1"+i])
        else:
            hist_df["mer1"+i] = tuika[:,j]
            hist_df["mer1"+i] = hist_df["mer1"+i].astype(np.float32)
        aggs["mer1"+i] = ["sum","mean","min","max"]
    # datetime features
    hist_df['purchase_date'] = pd.to_datetime(hist_df['purchase_date'])

    
    hist_df = reduce_mem_usage(hist_df)


    
    hist_df = hist_df.reset_index().groupby('card_id').agg(aggs)

    # change column name
    hist_df.columns = pd.Index([e[0] + "_" + e[1] for e in hist_df.columns.tolist()])
    hist_df.columns = ['new_'+ c for c in hist_df.columns]

    # reduce memory usage
    hist_df = reduce_mem_usage(hist_df)

    return hist_df