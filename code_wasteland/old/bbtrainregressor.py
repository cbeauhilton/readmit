import sys

sys.path.append("modules")
from datetime import datetime

startTime = datetime.now()


import lightgbm as lgb
import pandas as pd

try:
    import cPickle as pickle
except BaseException:
    import pickle
import config

# import configdocker as config

seed = config.SEED

print("Loading labels and features...")
r_train_labels = pd.read_pickle(config.R_TRAIN_LABELS_FILE)
r_train_features = pd.read_pickle(config.R_TRAIN_FEATURES_FILE)
r_test_labels = pd.read_pickle(config.R_TEST_LABELS_FILE)
r_test_features = pd.read_pickle(config.R_TEST_FEATURES_FILE)
r_valid_labels = pd.read_pickle(config.R_VALID_LABELS_FILE)
r_valid_features = pd.read_pickle(config.R_VALID_FEATURES_FILE)
r_features = pd.read_pickle(config.R_FEATURES_FILE)

print("Loading LightGBM datasets...")
r_d_train = lgb.Dataset(config.R_LIGHTGBM_READMIT_TRAIN_00)
r_d_test = lgb.Dataset(config.R_LIGHTGBM_READMIT_TEST_00)
r_d_valid = lgb.Dataset(config.R_LIGHTGBM_READMIT_VALID_00)

print("Training...")
r_evals_result = {}  # to record eval results for plotting


r_params = config.R_READMIT_PARAMS_LGBM

r_gbm = lgb.train(r_params, 
                r_d_train, 
                100000, 
                valid_sets=[r_d_test], 
                early_stopping_rounds=500,
                verbose_eval=5,
                evals_result=r_evals_result,
                keep_training_booster=True,
               )

print("Dumping model with pickle...")
pkl_model = config.LGBM_READMIT_MODEL_REGRESSION_PICKLE
with open(pkl_model, "wb") as fout:
    pickle.dump(r_gbm, fout)


# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")