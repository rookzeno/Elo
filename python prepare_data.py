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

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import reduce_mem_usage,rmse
from FE.FE import train_test,historical_transactions,new_merchant_transactions,additional_features
from FE.FE_target_encoding import target_encoding_hist,target_encoding_new
from FE.FE_observation_date_targetencoding import date_targetencoding
from FE.FE_monthlag_feature import historical_transactions_monthlag
from FE.feature_selection import feature_select
from FE.FE_monthly_change import monthly_change
from FE.FE_umap import umap
from FE.make_merchant_ID_usenew import make_new_mer
from FE.make_merchant_ID_usehist import make_hist_mer
from FE.hist_mer_new_mer import hist_mer_new_mer
from FE.hist_new_mer_histmer_newmer import hist_new_new_mer,hist_new_hist_mer
from FE.FE_month_data import hist_month
from FE.FE_denoy_amount import new_deanonimize,hist_deanonimize

debug = False
check = 1000 if debug else None
print("____________________Feture Engineering____________________")
df = train_test(check).reset_index()
df.to_feather("./feature/traintest.feather")
df = pd.merge(df,historical_transactions(check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,new_merchant_transactions(check).reset_index(),on="card_id",how="outer")
df = additional_features(df)
print("____________________target encoding____________________")
df = pd.merge(df,target_encoding_hist(check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,target_encoding_new(check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,date_targetencoding(check).reset_index(),on="card_id",how="outer")
print("____________________monthlag feture____________________")
df = pd.merge(df,historical_transactions_monthlag(0,check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,historical_transactions_monthlag(-2,check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,historical_transactions_monthlag(-6,check).reset_index(),on="card_id",how="outer")
print("____________________monthly change____________________")
df = pd.merge(df,monthly_change(check),on="card_id",how="outer")
print("____________________umap____________________")
df = pd.merge(df,umap(check),on="card_id",how="outer")
df.to_feather("./feature/prepare_data2.feather")
print("____________________feture selection____________________")
dfselect = feature_select(df)
dfselect.to_feather("./feature/selected.feather")
print("____________________make merchant ID Data____________________")
make_new_mer(check)
make_hist_mer(check)
print("____________________apply merchant ID Data____________________")
hist_mer_new_mer()
df = pd.merge(df,hist_new_new_mer(check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,hist_new_hist_mer(check).reset_index(),on="card_id",how="outer")
df.to_feather("./feature/prepare_data3.feather")
print("____________________feture selection2____________________")
dfselect = feature_select(df)
dfselect.to_feather("./feature/selected2.feather")
print("____________________month_data____________________")
df = pd.merge(df,hist_month(check).reset_index(),on="card_id",how="outer")
print("____________________deanonimize_data____________________")
df = pd.merge(df,new_deanonimize(check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,hist_deanonimize(-7,-3,check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,hist_deanonimize(-1,100,check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,hist_deanonimize(-3,100,check).reset_index(),on="card_id",how="outer")
df = pd.merge(df,hist_deanonimize(-100,100,check).reset_index(),on="card_id",how="outer")
df.to_feather("./feature/prepare_data4.feather")
print("____________________feture selection3____________________")
dfselect = feature_select(df)
dfselect.to_feather("./feature/selected3.feather")
print("____________________Finish____________________")