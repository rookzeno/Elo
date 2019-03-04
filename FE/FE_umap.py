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

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from utils import reduce_mem_usage,rmse,one_hot_encoder
from scipy.sparse.csgraph import connected_components

def umap(debug = False):
    df = feather.read_dataframe("./feature/traintest.feather")
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]
    if debug:
        train_df = train_df.iloc[:debug]
        test_df = test_df.iloc[:debug]
    train_len = train_df.shape[0]

    FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                    'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                    'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                    'OOF_PRED', 'month_0']
    cols = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    data = df[cols]
    outliers = train_df.outliers
    target = train_df.target

    #nan mean imputer:
    from sklearn.preprocessing import Imputer

    #from sklearn.impute import SimpleImputer
    imp = Imputer(missing_values=np.nan, strategy='mean')
    data = imp.fit_transform(data)
    #scale:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data=scaler.fit_transform(data)
    import umap
    from scipy.sparse.csgraph import connected_components

    embeddings = umap.UMAP().fit_transform(data)
    df3 = pd.DataFrame(index=df.index)
    df3["card_id"] = df.card_id
    df3["umap0"] = embeddings[:,0]
    df3["umap1"] = embeddings[:,1]
    df3["umap+umap"] = embeddings[:,0]+embeddings[:,1]
    df3["umap*umap"] = embeddings[:,0]*embeddings[:,1]
    return df3