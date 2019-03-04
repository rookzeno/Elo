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

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def hist_mer_new_mer():
    merchant = feather.read_dataframe("./input/hist_merchant.feather")
    merchant.set_index("merchant_id",inplace=True)
    merchant1 = feather.read_dataframe("./input/new_merchant.feather")
    merchant1.set_index("merchant_id",inplace=True)

    df = pd.merge(merchant1, merchant, on='merchant_id', how='outer')

    his = df.columns[78:]
    new = df.columns[:78]
    his = list(his)
    new = list(new)
    his.sort()
    new.sort()
    new.remove("hist_subsector_id_nunique")
    his.append("hist_subsector_id_nunique")
    his.remove("hist_purchase_date_max")
    his.remove("hist_purchase_date_min")
    new.remove("new_purchase_date_max")
    new.remove("new_purchase_date_min")

    df = df.fillna(0)
    df[his] = df[his].values+df[new].values
    hismer = df[his].rename(columns=lambda s: "mer+his" + s)
    hismer.reset_index().to_feather("./input/newmer+hist_merchant.feather")
