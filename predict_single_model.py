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

df = feather.read_dataframe("./feature/selected.feather")

feats = list(df.columns)
feats.remove("outliers")
feats.remove("target")
feats.remove("card_id")

train_df = df[df['target'].notnull()]
test_df = df[df['target'].isnull()]
train_df['rounded_target'] = train_df['target'].round(0)
train_df = train_df.sort_values('rounded_target').reset_index(drop=True)
vc = train_df['rounded_target'].value_counts()
vc = dict(sorted(vc.items()))
df1 = pd.DataFrame()
train_df['indexcol'],i = 0,1
for k,v in vc.items():
    step = train_df.shape[0]/v
    indent = train_df.shape[0]/(v+1)
    df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
    for j in range(0, v):
        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
    df1 = pd.concat([df2,df1])
    i+=1
train_df = df1.sort_values('indexcol', ascending=True).reset_index(drop=True)
del train_df['indexcol'], train_df['rounded_target']

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']
folds = KFold(n_splits= 6, shuffle=False, random_state=326)
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

    # set data structure
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(valid_x,
                           label=valid_y,
                           free_raw_data=False)

    # params optimized by optuna
    params ={
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9855232997390695,
            'max_depth': 7,
            'top_rate': 0.9064148448434349,
            'num_leaves': 63,
            'min_child_weight': 11.9612869171337,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 5.677537745007898,
            'colsample_bytree': 0.5665320670155495,
            'min_split_gain': 6.820197773625843,
            'reg_lambda': 6.2532317400459,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed':int(2**n_fold),
            'bagging_seed':int(2**n_fold),
            'drop_seed':int(2**n_fold)
            }

    reg = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    valid_names=['train', 'test'],
                    num_boost_round=10000,
                    early_stopping_rounds= 300,
                    verbose_eval=100
                    )
    oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"+str(n_fold)] = feats
    fold_importance_df["importance"+str(n_fold)] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
    fold_importance_df["fold"+str(n_fold)] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=1)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
    ans += rmse(valid_y, oof_preds[valid_idx])
print(ans/6)

test_df.loc[:,'target'] =sub_preds
test_df = test_df.reset_index(drop=True)
test_df[['card_id', 'target']].to_csv("./single_model.csv", index=False)