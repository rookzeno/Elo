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

from utils import rmse

first_preds = feather.read_dataframe("./stack/train_stack.feather")
first_test = feather.read_dataframe("./stack/test_stack.feather")
chan = {}
j = 0
for i in first_preds.columns:
    if i == "card_id": continue
    chan[i] = str(i+str(j))
    j += 1
first_preds = first_preds.rename(columns=chan)
first_test = first_test.rename(columns=chan)


first_preds = pd.merge(first_preds, feather.read_dataframe("./stack/train_stack1.feather"), on='card_id', how='outer')
first_test = pd.merge(first_test, feather.read_dataframe("./stack/test_stack1.feather"), left_index=True, right_index=True)

chan = {}
for i in first_preds.columns:
    if i == "card_id": continue
    chan[i] = str(i+str(j))
    j += 1
first_preds = first_preds.rename(columns=chan)
first_test = first_test.rename(columns=chan)

first_preds = pd.merge(first_preds, feather.read_dataframe("./stack/train_stack2.feather"), on='card_id', how='outer')
first_test = pd.merge(first_test, feather.read_dataframe("./stack/test_stack2.feather"), left_index=True, right_index=True)

chan = {}
for i in first_preds.columns:
    if i == "card_id": continue
    chan[i] = str(i+str(j))
    j += 1
first_preds = first_preds.rename(columns=chan)
first_test = first_test.rename(columns=chan)


train_card = first_preds.card_id
first_preds.drop("card_id",axis=1,inplace=True)

first_preds.fillna(0,inplace = True)
first_test.fillna(0,inplace=True)

feats = first_preds.columns
feats2 = []
for i in range(len(feats)):
    if feats[i].find("target") == -1 and  feats[i] != "card_id" and feats[i].find("outliers") == -1:
        feats2.append(feats[i])
    elif feats[i].find("target") != -1:
        t_name= feats[i]
feats = feats2

first_preds = first_preds[feats + [t_name]]
first_test = first_test[feats]

#standardize
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
first = sc.fit_transform(first_preds[feats])
train_df2 = pd.DataFrame(first, index=first_preds.index, columns=feats)
train_df2[t_name] = first_preds[t_name]
test = sc.transform(first_test[feats])
test_df2 = pd.DataFrame(test, index=first_test.index, columns=feats)
first_preds = train_df2
first_test = test_df2
first_preds["card_id"] = train_card

from sklearn.linear_model import Ridge
rid = Ridge()

def build_model(dropout_rate=0.25,activation='relu'):
    start_neurons = 512
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

first_preds = first_preds.sort_values("card_id")
final_preds = pd.DataFrame(index=first_preds.index)
final_preds["target"] = first_preds[t_name] 
final_test = pd.DataFrame(index=first_test.index)
final_preds["card_id"] = first_preds.card_id
test = pd.read_csv("./input/test.csv")
final_test["card_id"] = test.card_id
final_preds["preds"] = 0
final_test["target"] = 0
for counts in range(3):
    folds = KFold(n_splits= 6, shuffle=False, random_state=326)
    first_preds['rounded_target'] = first_preds[t_name].round(0)
    first_preds = first_preds.sort_values('rounded_target').reset_index(drop=True)
    vc = first_preds['rounded_target'].value_counts()
    vc = dict(sorted(vc.items()))
    df = pd.DataFrame()
    first_preds['indexcol'],i = 0,1
    for k,v in vc.items():
        step = first_preds.shape[0]/v
        indent = first_preds.shape[0]/(v+1)
        df2 = first_preds[first_preds['rounded_target'] == k].sample(v, random_state=10835+counts).reset_index(drop=True)
        for j in range(0, v):
            df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
        df = pd.concat([df2,df])
        i+=1
    first_preds = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
    del first_preds['indexcol'], first_preds['rounded_target']
    second_preds = pd.DataFrame(index=first_preds.index)
    second_preds["target"] = first_preds[t_name] 
    second_test = pd.DataFrame(index=first_test.index)
    second_preds["card_id"] = first_preds["card_id"]
    oof_preds = np.zeros(first_preds.shape[0])
    sub_preds = np.zeros(first_test.shape[0])
    feature_importance_df = pd.DataFrame()
    ans = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(first_preds[feats])):
        train_x, train_y = first_preds[feats].iloc[train_idx], first_preds[t_name].iloc[train_idx]
        valid_x, valid_y = first_preds[feats].iloc[valid_idx], first_preds[t_name].iloc[valid_idx]
        model = build_model()    
        model.compile(loss="mean_squared_error", optimizer='adam')
        early_stopping = EarlyStopping(monitor='val_loss', mode = 'min',patience=18, verbose=1)
        model_checkpoint = ModelCheckpoint("./nnnnnyou.model",monitor='val_loss', mode = 'min', save_best_only=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=6, min_lr=0.0001, verbose=0)
        history = model.fit(train_x, train_y,
                            validation_data=( valid_x,valid_y), 
                            epochs=50,
                            batch_size=512,
                            callbacks=[model_checkpoint, reduce_lr,early_stopping], 
                            verbose=0)
        model = load_model("./nnnnnyou.model")
        pred_y = model.predict(valid_x).reshape(-1)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
        oof_preds[valid_idx] = pred_y
        sub_preds += model.predict(first_test[feats]).reshape(-1) / folds.n_splits
        ans += rmse(valid_y, pred_y)
    second_preds["nn"] = oof_preds
    second_test["nn"] = sub_preds
    print(ans/6)
    ans = 0
    oof_preds = np.zeros(first_preds.shape[0])
    sub_preds = np.zeros(first_test.shape[0])
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(first_preds[feats])):
        train_x, train_y = first_preds[feats].iloc[train_idx], first_preds[t_name].iloc[train_idx]
        valid_x, valid_y = first_preds[feats].iloc[valid_idx], first_preds[t_name].iloc[valid_idx]
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)
        rid.fit(train_x,train_y)
        pred_y = rid.predict(valid_x)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, pred_y)))
        oof_preds[valid_idx] = pred_y
        ans += rmse(valid_y, pred_y)
        sub_preds += rid.predict(first_test[feats]) / folds.n_splits
    second_preds["ridge"] = oof_preds
    second_test["ridge"] = sub_preds
    print(ans/folds.n_splits)
    oof_preds = np.zeros(first_preds.shape[0])
    sub_preds = np.zeros(first_test.shape[0])
    ans = 0
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(first_preds[feats])):
        train_x, train_y = first_preds[feats].iloc[train_idx], first_preds[t_name].iloc[train_idx]
        valid_x, valid_y = first_preds[feats].iloc[valid_idx], first_preds[t_name].iloc[valid_idx]
        params = {'num_leaves': 1024,
             'min_data_in_leaf': 30, 
             'objective':'regression',
             'max_depth': 3,
             'learning_rate': 0.01,
             "min_child_samples": 20,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.1,
             "verbosity": -1,
             "nthread": 4,
             "random_state": 4590}
    
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
                        early_stopping_rounds= 200,
                        verbose_eval=0
                        )
        pred_y = reg.predict(valid_x, num_iteration=reg.best_iteration)
        pred_y[np.isnan(pred_y)] = 0
        oof_preds[valid_idx] = pred_y
        ans += rmse(valid_y, oof_preds[valid_idx])
        sub_preds += reg.predict(first_test[feats], num_iteration=reg.best_iteration) / folds.n_splits
    second_preds["lgb"] = oof_preds
    second_test["lgb"] = sub_preds
    print(ans/folds.n_splits)
    second_preds = second_preds.sort_values("card_id")
    final_preds["preds"] += list((second_preds["ridge"]*0.3 + second_preds["nn"]*0.6 + second_preds["lgb"]*0.1)/3)
    final_test["target"] += list((second_test["ridge"]*0.3 + second_test["nn"]*0.6 + second_test["lgb"]*0.1)/3)
final_preds.to_csv("train_Finish.csv",index=False)
final_test.to_csv("test_Finish.csv",index=False)
print(rmse(final_preds.target,final_preds.preds))