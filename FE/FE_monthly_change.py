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

def monthly_change(num_rows=None):
    hist_df = pd.read_csv('./input/historical_transactions.csv', nrows=num_rows)
    hist_df['purchase_amount'] = np.round(hist_df['purchase_amount'] / 0.00150265118 + 497.06,8)
    df = feather.read_dataframe("./feature/traintest.feather")
    irisou = hist_df.groupby(["card_id","month_lag"])['purchase_amount'].agg(["sum"])
    rese = irisou.reset_index()
    card_id = list(rese.card_id.unique())
    lag = sorted(list(hist_df.month_lag.unique()),key=lambda x : -x)
    konkai = pd.DataFrame(index = card_id,columns=lag)
    for i in konkai.index:
        for j,k in zip(irisou.loc[i].index,irisou.loc[i].values):
            konkai.loc[i][j] = k[0]
    hist_df = pd.read_csv('./input/new_merchant_transactions.csv', nrows=num_rows)
    hist_df['purchase_amount'] = np.round(hist_df['purchase_amount'] / 0.00150265118 + 497.06,8)
    irisou = hist_df.groupby(["card_id","month_lag"])['purchase_amount'].agg(["sum"])
    rese = irisou.reset_index()
    card_id = list(rese.card_id.unique())
    lag = sorted(list(hist_df.month_lag.unique()),key=lambda x : -x)
    kon = pd.DataFrame(index = card_id,columns=lag)
    konkai.index.names = ['card_id']
    kon.index.names = ['card_id']
    for i in kon.index:
        for j,k in zip(irisou.loc[i].index,irisou.loc[i].values):
            kon.loc[i][j] = k[0]
    henka = pd.merge(konkai,kon,on="card_id",how="outer")
    henka = henka.sort_index(axis=1, ascending=False)
    henka.fillna(0,inplace = True)
    for i in range(-12,3):
        henka[i] = henka[i] + henka[i-1]
    ima = henka.columns
    henka["month_lag_mean"] = henka[ima].mean(axis=1)
    henka["month_lag_sum"] = henka[ima].sum(axis=1)
    henka["month_lag_min"] = henka[ima][henka[ima] != 0].min(axis=1)
    henka["month_lag_max"] = henka[ima].max(axis=1)
    henka["month_lag_var"] = henka[ima].var(axis=1)
    henka["month_lag_maxindex"] = henka[ima].idxmax(axis=1)
    henka["month_lag_minindex"] = henka[ima].idxmin(axis=1)
    henka["new_%"] = (henka[1] + henka[2])/(henka["month_lag_sum"]-henka[1] - henka[2])
    for i in ima:
        henka[str(i) + "%"] = henka[i]/henka["month_lag_sum"]
    henka["2-1"] = henka[2]-henka[1]
    henka["1-0"] = henka[1]-henka[0]
    henka["0--1"] = henka[0]-henka[-1]
    henka["-1--2"] = henka[-1]-henka[-2]
    henka["-2--3"] = henka[-2]-henka[-3]
    henka["-3--4"] = henka[-3]-henka[-4]
    henka["-4--5"] = henka[-4]-henka[-2]
    henka["-5--6"] = henka[-5]-henka[-6]
    henka["-6--7"] = henka[-6]-henka[-7]
    henka["-7--8"] = henka[-7]-henka[-8]
    henka["-8--9"] = henka[-8]-henka[-9]
    henka["-9--10"] = henka[-9]-henka[-10]
    henka["-10--11"] = henka[-10]-henka[-11]
    henka["-11--12"] = henka[-11]-henka[-12]
    henka["-12--13"] = henka[-12]-henka[-13]
    henka.columns = ['henka2_'+ str(c) for c in henka.columns]
    return henka