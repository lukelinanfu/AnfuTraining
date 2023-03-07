from sklearn.metrics import mean_absolute_percentage_error
from hyperopt import fmin, tpe, hp, anneal, Trials

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from functools import partial

import pandas as pd
import os
from pathlib import Path
import sys

def split_out_validation_by_time(df, start_time_num):
  """
  Pass a df, return a new df to build a new validation set.
  The mew validation set starts from the `start_time_num`.

  """
  print(df['raw_time'], df['raw_time'].shape)
  return df.loc[df['raw_time'] >= start_time_num]

def train_model_and_optimize_loss(paras, data):
  
  init = paras["init"]
  min_samples_split = paras["min_samples_split"]
  learning_rate = paras["learning_rate"]
  max_depth = paras["max_depth"]
  min_samples_leaf = paras["min_samples_leaf"]
  n_estimators = paras["n_estimators"]

  print(init, min_samples_split, learning_rate, max_depth, min_samples_leaf, n_estimators,
        type(init), type(min_samples_split), type(learning_rate), type(max_depth), type(min_samples_leaf), type(n_estimators))
  
  X_train, Y_train, X_test, Y_test = data
  X_validation = split_out_validation_by_time(X_train, 11001).drop(columns=['raw_time'])
  Y_validation = split_out_validation_by_time(Y_train, 11001).drop(columns=['raw_time'])
  
  # 訓練前要去掉raw_time
  X_train = X_train.drop(columns=['raw_time'])
  X_test = X_test.drop(columns=['raw_time'])
  Y_train = Y_train.drop(columns=['raw_time'])
  Y_test = Y_test.drop(columns=['raw_time'])
  
  # Use these params combinations to train a model
  if init == "True":
    init_estimator = Ridge()
    init_estimator.fit(X_train, Y_train)

  else:
    init_estimator = None

  gbm = GradientBoostingRegressor(init=init_estimator,
                  min_samples_split=min_samples_split,
                  learning_rate=learning_rate,
                  max_depth=max_depth,
                  min_samples_leaf=min_samples_leaf,
                  max_features='sqrt',
                  loss='huber',
                  subsample=0.8,
                  random_state=50,
                  n_estimators=n_estimators,
                  verbose=1)


  gbm.fit(X_train, Y_train)
  print("After training")
  mape = mean_absolute_percentage_error(Y_validation, gbm.predict(X_validation) )

  history_record = {
    'loss': mape,
    'init': init,
    'min_samples_split': min_samples_split,
    'learning_rate': learning_rate,
    'max_depth': max_depth,
    'min_samples_leaf': min_samples_leaf,
    'n_estimators': n_estimators
  }
  
  history = pd.DataFrame(history_record, index=[0])
  if not os.path.exists('./history_tuning/use_better_validation'):
    os.mkdir('./history_tuning/use_better_validation')

  if Path(f'./history_tuning/use_better_validation/{county_name}result.csv').exists():
    history = pd.concat([pd.read_csv(f'./history_tuning/use_better_validation/{county_name}result.csv'), history])
  
  history.to_csv(f'./history_tuning/use_better_validation/{county_name}result.csv', index=False)
  

  return mape

PATH = '.\\data'
# Read train and testing file

def read_df(county_name):
    X_train = pd.read_csv(Path(PATH)/f"{county_name}_X_train.csv")
    Y_train = pd.read_csv(Path(PATH)/f"{county_name}_Y_train.csv")
    X_test = pd.read_csv(Path(PATH)/f"{county_name}_X_test.csv")
    Y_test = pd.read_csv(Path(PATH)/f"{county_name}_Y_test.csv")

    return X_train, Y_train, X_test, Y_test



county_name = sys.argv[1]
X_train, Y_train, X_test, Y_test = read_df(county_name)
print(county_name, X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)



# 調參數區
import numpy as np
parameter_space = {
  'init': hp.choice('init', ['False', 'True']),
  'min_samples_split': hp.uniform('min_samples_split', 0, 1),
  'learning_rate': hp.uniform('learning_rate', 0, 0.5),
  'max_depth': hp.choice('max_depth', np.arange(1, 1000+1, dtype=int)), # should be int
  'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 1000+1, dtype=int)),
  'n_estimators': hp.choice('n_estimators', np.arange(1, 1000+1, dtype=int)),
}

trials = Trials()
fmin_objective = partial(train_model_and_optimize_loss, data=[X_train, Y_train, X_test, Y_test])
min_w = fmin(fn=fmin_objective, space=parameter_space, algo=tpe.suggest, max_evals=1500, trials=trials)