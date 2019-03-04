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
import math

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm_notebook as tqdm
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import feather
from utils import rmse

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
df = pd.DataFrame()
train_df['indexcol'],i = 0,1
for k,v in vc.items():
    step = train_df.shape[0]/v
    indent = train_df.shape[0]/(v+1)
    df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=40).reset_index(drop=True)
    for j in range(0, v):
        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
    df = pd.concat([df2,df])
    i+=1
train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
del train_df['indexcol'], train_df['rounded_target']

from sklearn.linear_model import Ridge
import xgboost as xgb
xgbr = xgb.XGBRegressor(colsample_bytree=0.1, colsample_bylevel =0.5, 
                             gamma=2, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=10, n_estimators=1000, reg_alpha=1, 
                             reg_lambda = 1,eval_metric = 'rmse', subsample=0.8, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =4, nthread = -1)


folds = KFold(n_splits= 6, shuffle=False, random_state=326)
first_preds = pd.DataFrame(index=train_df.index)
first_preds["target"] = train_df["target"] 
first_test = pd.DataFrame(index=test_df.index)


#predict xgb
sub_preds = np.zeros(test_df.shape[0])
oof_preds = np.zeros(train_df.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]
    xgbr.fit(train_x,train_y)
    pred_y = xgbr.predict(valid_x)
    pred_y[np.isnan(pred_y)] = 0
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += xgbr.predict(test_df[feats]) / folds.n_splits
first_preds["xgb"] = oof_preds
first_test["xgb"] = sub_preds


#predict lgb
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

    params = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 459}
    
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(valid_x,
                           label=valid_y,
                           free_raw_data=False)
    reg = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    valid_names=['train', 'test'],
                    num_boost_round=10000,
                    early_stopping_rounds= 600,
                    verbose_eval=100
                    )
    pred_y = reg.predict(valid_x, num_iteration=reg.best_iteration)
    pred_y[np.isnan(pred_y)] = 0
    oof_preds[valid_idx] = pred_y
    ans += rmse(valid_y, oof_preds[valid_idx])
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits
first_preds["lgb"] = oof_preds
first_test["lgb"] = sub_preds
print(ans/ folds.n_splits)

#predict lgb2
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

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
    
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(valid_x,
                           label=valid_y,
                           free_raw_data=False)
    reg = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    valid_names=['train', 'test'],
                    num_boost_round=10000,
                    early_stopping_rounds= 600,
                    verbose_eval=100
                    )
    pred_y = reg.predict(valid_x, num_iteration=reg.best_iteration)
    pred_y[np.isnan(pred_y)] = 0
    oof_preds[valid_idx] = pred_y
    ans += rmse(valid_y, oof_preds[valid_idx])
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits
first_preds["lgb2"] = oof_preds
first_test["lgb2"] = sub_preds
print(ans/ folds.n_splits)

#predict catboost
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=700,
                             learning_rate=0.025,
                             depth=12,
                             eval_metric='RMSE',
                             random_seed = 25,
                             bagging_temperature = 0.3,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
sub_preds = np.zeros(test_df.shape[0])
oof_preds = np.zeros(train_df.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]
    cb_model.fit(train_x, train_y,
             eval_set=(valid_x,valid_y),
             use_best_model=True,
             )
    pred_y = cb_model.predict(valid_x)
    pred_y[np.isnan(pred_y)] = 0
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += cb_model.predict(test_df[feats]) / folds.n_splits
first_preds["cat"] = oof_preds
first_test["cat"] = sub_preds

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from keras import regularizers
from collections import Counter
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.layers import PReLU



def build_model(dropout_rate=0.25,activation='relu'):
    start_neurons = 1024
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=len(feats),kernel_initializer='he_normal',))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//4,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//8,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//16,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    
    model.add(Dense(1, activation='linear'))
    return model

#standardize
test_card = test_df.card_id
train_card = train_df.card_id
test_df.drop("card_id",axis=1,inplace=True)
train_df.drop("card_id",axis=1,inplace=True)
for i in train_df.columns:
    train_df[i].replace(float("inf"),train_df[i][train_df[i] != float("inf")].max()+1,inplace=True)
    train_df[i].replace(-float("inf"),train_df[i][train_df[i] != -float("inf")].min()-1,inplace=True)
    test_df[i].replace(float("inf"),train_df[i][train_df[i] != float("inf")].max(),inplace=True)
    test_df[i].replace(-float("inf"),train_df[i][train_df[i] != -float("inf")].min(),inplace=True)
test_df.fillna(train_df.mean(),inplace = True)
train_df.fillna(train_df.mean(),inplace = True)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train = sc.fit_transform(train_df)
train_df2 = pd.DataFrame(train, index=train_df.index, columns=train_df.columns)
train_df2["target"] = train_df["target"]
test = sc.transform(test_df)
test_df2 = pd.DataFrame(test, index=test_df.index, columns=test_df.columns)
test_df2["target"] = test_df["target"]
test_df2["card_id"] = train_card
train_df2["card_id"] = test_card

#predict NN
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    model = build_model(dropout_rate=0.4)
    model.compile(loss="mean_squared_error", optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', mode = 'min',patience=15, verbose=1)
    model_checkpoint = ModelCheckpoint("./nnnyou.model",monitor='val_loss', mode = 'min', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=6, min_lr=0.0001, verbose=1)
    history = model.fit(train_x, train_y,
                        validation_data=( valid_x,valid_y), 
                        epochs=50,
                        batch_size=512,
                        callbacks=[model_checkpoint, reduce_lr,early_stopping], 
                        verbose=2)
    model = load_model("./nnnyou.model")
    pred_y = model.predict(valid_x).reshape(-1)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += model.predict(test_df2[feats]).reshape(-1) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["nn"] = oof_preds
first_test["nn"] = sub_preds
print(ans/ folds.n_splits)

#predict randomforest
from sklearn.ensemble import RandomForestRegressor
reg = lgb.LGBMRegressor(boosting_type="rf",
                 num_leaves=1024,
                 max_depth=6,
                 n_estimators=500,  # 1000
                 subsample=.623,  # .623
                 colsample_bytree=.5,
                 bagging_freq = 3     )  # .5
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    reg.fit(train_x, train_y)
    pred_y = reg.predict(valid_x)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += reg.predict(test_df2[feats]) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["random"] = oof_preds
first_test["random"] = sub_preds
print(ans/ folds.n_splits)

#predict ridge
from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1)
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    reg.fit(train_x, train_y)
    pred_y = reg.predict(valid_x)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += reg.predict(test_df2[feats]) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["ridgek"] = oof_preds
first_test["ridgek"] = sub_preds
print(ans/ folds.n_splits)

first_preds["card_id"] = train_card
first_preds.to_feather("./stack/train_stack.feather")
first_test.reset_index().to_feather("./stack/test_stack.feather")

df = feather.read_dataframe("./feature/selected2.feather")
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
df = pd.DataFrame()
train_df['indexcol'],i = 0,1
for k,v in vc.items():
    step = train_df.shape[0]/v
    indent = train_df.shape[0]/(v+1)
    df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=400).reset_index(drop=True)
    for j in range(0, v):
        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
    df = pd.concat([df2,df])
    i+=1
train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
del train_df['indexcol'], train_df['rounded_target']

from sklearn.linear_model import Ridge
import xgboost as xgb
xgbr = xgb.XGBRegressor(colsample_bytree=0.1, colsample_bylevel =0.5, 
                             gamma=2, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=10, n_estimators=1000, reg_alpha=1, 
                             reg_lambda = 1,eval_metric = 'rmse', subsample=0.8, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =4, nthread = -1)


folds = KFold(n_splits= 6, shuffle=False, random_state=326)
first_preds = pd.DataFrame(index=train_df.index)
first_preds["target"] = train_df["target"] 
first_test = pd.DataFrame(index=test_df.index)


#predict xgb
sub_preds = np.zeros(test_df.shape[0])
oof_preds = np.zeros(train_df.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]
    xgbr.fit(train_x,train_y)
    pred_y = xgbr.predict(valid_x)
    pred_y[np.isnan(pred_y)] = 0
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += xgbr.predict(test_df[feats]) / folds.n_splits
first_preds["xgb"] = oof_preds
first_test["xgb"] = sub_preds


#predict lgb
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

    params = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 459}
    
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(valid_x,
                           label=valid_y,
                           free_raw_data=False)
    reg = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    valid_names=['train', 'test'],
                    num_boost_round=10000,
                    early_stopping_rounds= 600,
                    verbose_eval=100
                    )
    pred_y = reg.predict(valid_x, num_iteration=reg.best_iteration)
    pred_y[np.isnan(pred_y)] = 0
    oof_preds[valid_idx] = pred_y
    ans += rmse(valid_y, oof_preds[valid_idx])
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits
first_preds["lgb"] = oof_preds
first_test["lgb"] = sub_preds
print(ans/ folds.n_splits)

#predict lgb2
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

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
    
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(valid_x,
                           label=valid_y,
                           free_raw_data=False)
    reg = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    valid_names=['train', 'test'],
                    num_boost_round=10000,
                    early_stopping_rounds= 600,
                    verbose_eval=100
                    )
    pred_y = reg.predict(valid_x, num_iteration=reg.best_iteration)
    pred_y[np.isnan(pred_y)] = 0
    oof_preds[valid_idx] = pred_y
    ans += rmse(valid_y, oof_preds[valid_idx])
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits
first_preds["lgb2"] = oof_preds
first_test["lgb2"] = sub_preds
print(ans/ folds.n_splits)

#predict catboost
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=700,
                             learning_rate=0.025,
                             depth=12,
                             eval_metric='RMSE',
                             random_seed = 25,
                             bagging_temperature = 0.3,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
sub_preds = np.zeros(test_df.shape[0])
oof_preds = np.zeros(train_df.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]
    cb_model.fit(train_x, train_y,
             eval_set=(valid_x,valid_y),
             use_best_model=True,
             )
    pred_y = cb_model.predict(valid_x)
    pred_y[np.isnan(pred_y)] = 0
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += cb_model.predict(test_df[feats]) / folds.n_splits
first_preds["cat"] = oof_preds
first_test["cat"] = sub_preds

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from keras import regularizers
from collections import Counter
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.layers import PReLU



def build_model(dropout_rate=0.25,activation='relu'):
    start_neurons = 1024
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=len(feats),kernel_initializer='he_normal',))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//4,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//8,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//16,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    
    model.add(Dense(1, activation='linear'))
    return model

#standardize
test_card = test_df.card_id
train_card = train_df.card_id
test_df.drop("card_id",axis=1,inplace=True)
train_df.drop("card_id",axis=1,inplace=True)
for i in train_df.columns:
    train_df[i].replace(float("inf"),train_df[i][train_df[i] != float("inf")].max()+1,inplace=True)
    train_df[i].replace(-float("inf"),train_df[i][train_df[i] != -float("inf")].min()-1,inplace=True)
    test_df[i].replace(float("inf"),train_df[i][train_df[i] != float("inf")].max(),inplace=True)
    test_df[i].replace(-float("inf"),train_df[i][train_df[i] != -float("inf")].min(),inplace=True)
test_df.fillna(train_df.mean(),inplace = True)
train_df.fillna(train_df.mean(),inplace = True)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train = sc.fit_transform(train_df)
train_df2 = pd.DataFrame(train, index=train_df.index, columns=train_df.columns)
train_df2["target"] = train_df["target"]
test = sc.transform(test_df)
test_df2 = pd.DataFrame(test, index=test_df.index, columns=test_df.columns)
test_df2["target"] = test_df["target"]
test_df2["card_id"] = train_card
train_df2["card_id"] = test_card

#predict NN
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    model = build_model(dropout_rate=0.4)
    model.compile(loss="mean_squared_error", optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', mode = 'min',patience=15, verbose=1)
    model_checkpoint = ModelCheckpoint("./nnnyou.model",monitor='val_loss', mode = 'min', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=6, min_lr=0.0001, verbose=1)
    history = model.fit(train_x, train_y,
                        validation_data=( valid_x,valid_y), 
                        epochs=50,
                        batch_size=512,
                        callbacks=[model_checkpoint, reduce_lr,early_stopping], 
                        verbose=2)
    model = load_model("./nnnyou.model")
    pred_y = model.predict(valid_x).reshape(-1)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += model.predict(test_df2[feats]).reshape(-1) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["nn"] = oof_preds
first_test["nn"] = sub_preds
print(ans/ folds.n_splits)

#predict randomforest
from sklearn.ensemble import RandomForestRegressor
reg = lgb.LGBMRegressor(boosting_type="rf",
                 num_leaves=1024,
                 max_depth=6,
                 n_estimators=500,  # 1000
                 subsample=.623,  # .623
                 colsample_bytree=.5,
                 bagging_freq = 3     )  # .5
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    reg.fit(train_x, train_y)
    pred_y = reg.predict(valid_x)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += reg.predict(test_df2[feats]) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["random"] = oof_preds
first_test["random"] = sub_preds
print(ans/ folds.n_splits)

#predict ridge
from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1)
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    reg.fit(train_x, train_y)
    pred_y = reg.predict(valid_x)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += reg.predict(test_df2[feats]) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["ridgek"] = oof_preds
first_test["ridgek"] = sub_preds
print(ans/ folds.n_splits)

first_preds["card_id"] = train_card
first_preds.to_feather("./stack/train_stack1.feather")
first_test.reset_index().to_feather("./stack/test_stack1.feather")








df = feather.read_dataframe("./feature/selected3.feather")
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
df = pd.DataFrame()
train_df['indexcol'],i = 0,1
for k,v in vc.items():
    step = train_df.shape[0]/v
    indent = train_df.shape[0]/(v+1)
    df2 = train_df[train_df['rounded_target'] == k].sample(v, random_state=400).reset_index(drop=True)
    for j in range(0, v):
        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
    df = pd.concat([df2,df])
    i+=1
train_df = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
del train_df['indexcol'], train_df['rounded_target']

from sklearn.linear_model import Ridge
import xgboost as xgb
xgbr = xgb.XGBRegressor(colsample_bytree=0.1, colsample_bylevel =0.5, 
                             gamma=2, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=10, n_estimators=1000, reg_alpha=1, 
                             reg_lambda = 1,eval_metric = 'rmse', subsample=0.8, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =4, nthread = -1)


folds = KFold(n_splits= 6, shuffle=False, random_state=326)
first_preds = pd.DataFrame(index=train_df.index)
first_preds["target"] = train_df["target"] 
first_test = pd.DataFrame(index=test_df.index)


#predict xgb
sub_preds = np.zeros(test_df.shape[0])
oof_preds = np.zeros(train_df.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]
    xgbr.fit(train_x,train_y)
    pred_y = xgbr.predict(valid_x)
    pred_y[np.isnan(pred_y)] = 0
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += xgbr.predict(test_df[feats]) / folds.n_splits
first_preds["xgb"] = oof_preds
first_test["xgb"] = sub_preds


#predict lgb
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

    params = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 459}
    
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(valid_x,
                           label=valid_y,
                           free_raw_data=False)
    reg = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    valid_names=['train', 'test'],
                    num_boost_round=10000,
                    early_stopping_rounds= 600,
                    verbose_eval=100
                    )
    pred_y = reg.predict(valid_x, num_iteration=reg.best_iteration)
    pred_y[np.isnan(pred_y)] = 0
    oof_preds[valid_idx] = pred_y
    ans += rmse(valid_y, oof_preds[valid_idx])
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits
first_preds["lgb"] = oof_preds
first_test["lgb"] = sub_preds
print(ans/ folds.n_splits)

#predict lgb2
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

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
    
    lgb_train = lgb.Dataset(train_x,
                            label=train_y,
                            free_raw_data=False)
    lgb_test = lgb.Dataset(valid_x,
                           label=valid_y,
                           free_raw_data=False)
    reg = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    valid_names=['train', 'test'],
                    num_boost_round=10000,
                    early_stopping_rounds= 600,
                    verbose_eval=100
                    )
    pred_y = reg.predict(valid_x, num_iteration=reg.best_iteration)
    pred_y[np.isnan(pred_y)] = 0
    oof_preds[valid_idx] = pred_y
    ans += rmse(valid_y, oof_preds[valid_idx])
    sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits
first_preds["lgb2"] = oof_preds
first_test["lgb2"] = sub_preds
print(ans/ folds.n_splits)

#predict catboost
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(iterations=700,
                             learning_rate=0.025,
                             depth=12,
                             eval_metric='RMSE',
                             random_seed = 25,
                             bagging_temperature = 0.3,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
sub_preds = np.zeros(test_df.shape[0])
oof_preds = np.zeros(train_df.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]
    cb_model.fit(train_x, train_y,
             eval_set=(valid_x,valid_y),
             use_best_model=True,
             )
    pred_y = cb_model.predict(valid_x)
    pred_y[np.isnan(pred_y)] = 0
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += cb_model.predict(test_df[feats]) / folds.n_splits
first_preds["cat"] = oof_preds
first_test["cat"] = sub_preds

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import keras
from keras import regularizers
from collections import Counter
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.layers import PReLU



def build_model(dropout_rate=0.25,activation='relu'):
    start_neurons = 1024
    # create model
    model = Sequential()
    model.add(Dense(start_neurons, input_dim=len(feats),kernel_initializer='he_normal',))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//4,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//8,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(start_neurons//16,kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    
    model.add(Dense(1, activation='linear'))
    return model

#standardize
test_card = test_df.card_id
train_card = train_df.card_id
test_df.drop("card_id",axis=1,inplace=True)
train_df.drop("card_id",axis=1,inplace=True)
for i in train_df.columns:
    train_df[i].replace(float("inf"),train_df[i][train_df[i] != float("inf")].max()+1,inplace=True)
    train_df[i].replace(-float("inf"),train_df[i][train_df[i] != -float("inf")].min()-1,inplace=True)
    test_df[i].replace(float("inf"),train_df[i][train_df[i] != float("inf")].max(),inplace=True)
    test_df[i].replace(-float("inf"),train_df[i][train_df[i] != -float("inf")].min(),inplace=True)
test_df.fillna(train_df.mean(),inplace = True)
train_df.fillna(train_df.mean(),inplace = True)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train = sc.fit_transform(train_df)
train_df2 = pd.DataFrame(train, index=train_df.index, columns=train_df.columns)
train_df2["target"] = train_df["target"]
test = sc.transform(test_df)
test_df2 = pd.DataFrame(test, index=test_df.index, columns=test_df.columns)
test_df2["target"] = test_df["target"]
test_df2["card_id"] = train_card
train_df2["card_id"] = test_card

#predict NN
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    model = build_model(dropout_rate=0.4)
    model.compile(loss="mean_squared_error", optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', mode = 'min',patience=15, verbose=1)
    model_checkpoint = ModelCheckpoint("./nnnyou.model",monitor='val_loss', mode = 'min', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=6, min_lr=0.0001, verbose=1)
    history = model.fit(train_x, train_y,
                        validation_data=( valid_x,valid_y), 
                        epochs=50,
                        batch_size=512,
                        callbacks=[model_checkpoint, reduce_lr,early_stopping], 
                        verbose=2)
    model = load_model("./nnnyou.model")
    pred_y = model.predict(valid_x).reshape(-1)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += model.predict(test_df2[feats]).reshape(-1) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["nn"] = oof_preds
first_test["nn"] = sub_preds
print(ans/ folds.n_splits)

#predict randomforest
from sklearn.ensemble import RandomForestRegressor
reg = lgb.LGBMRegressor(boosting_type="rf",
                 num_leaves=1024,
                 max_depth=6,
                 n_estimators=500,  # 1000
                 subsample=.623,  # .623
                 colsample_bytree=.5,
                 bagging_freq = 3     )  # .5
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    reg.fit(train_x, train_y)
    pred_y = reg.predict(valid_x)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += reg.predict(test_df2[feats]) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["random"] = oof_preds
first_test["random"] = sub_preds
print(ans/ folds.n_splits)

#predict ridge
from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1)
oof_preds = np.zeros(first_preds.shape[0])
sub_preds = np.zeros(first_test.shape[0])
feature_importance_df = pd.DataFrame()
ans = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df2[feats])):
    train_x, train_y = train_df2[feats].iloc[train_idx], train_df2["target"].iloc[train_idx]
    valid_x, valid_y = train_df2[feats].iloc[valid_idx], train_df2["target"].iloc[valid_idx]
    reg.fit(train_x, train_y)
    pred_y = reg.predict(valid_x)
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
    oof_preds[valid_idx] = pred_y
    sub_preds += reg.predict(test_df2[feats]) / folds.n_splits
    ans += rmse(valid_y, pred_y)
first_preds["ridgek"] = oof_preds
first_test["ridgek"] = sub_preds
print(ans/ folds.n_splits)

first_preds["card_id"] = train_card
first_preds.to_feather("./stack/train_stack2.feather")
first_test.reset_index().to_feather("./stack/test_stack2.feather")