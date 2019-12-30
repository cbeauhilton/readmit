import os
import time
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import UndefinedMetricWarning

import cbh.config as config
import cbh.configcols as configcols
from cbh.generalHelpers import (
    lgb_f1_score,
    make_datafolder_for_target,
    make_figfolder_for_target,
    make_modelfolder_for_target,
    make_report_tables_folder,
    train_test_valid_80_10_10_split,
    get_latest_folders
)
from cbh.lgbmHelpers import lgbmClassificationHelpers
from cbh.shapHelpers import shapHelpers

try:
    import cPickle as pickle
except BaseException:
    import pickle



print("About to run", os.path.basename(__file__))
startTime = datetime.now()
seed = config.SEED

target = "length_of_stay_over_5_days"
name_for_figs = "Length of Stay"

final_file = config.PROCESSED_DATA_DIR / "los5d.h5"
data = pd.read_hdf(final_file, key="los5dclean")

print("File loaded.")

debug = False
print("Debug:", debug)

if debug:
    data = data[:20000]

figfolder = make_figfolder_for_target(debug, target)
datafolder = make_datafolder_for_target(debug, target)
modelfolder = make_modelfolder_for_target(debug, target)
tablefolder = make_report_tables_folder(debug)

train_set, test_set, valid_set = train_test_valid_80_10_10_split(data, target, seed)

train_labels = train_set[target]
train_features = train_set.drop([target], axis=1)

test_labels = test_set[target]
test_features = test_set.drop([target], axis=1)

valid_labels = valid_set[target]
valid_features = valid_set.drop([target], axis=1)

labels = data[target]
features = data.drop([target], axis=1)


print("Dropping length of stay in days for LoS targets...")
features = features.drop(["length_of_stay_in_days"], axis=1)
train_features = train_features.drop(["length_of_stay_in_days"], axis=1)
test_features = test_features.drop(["length_of_stay_in_days"], axis=1)
valid_features = valid_features.drop(["length_of_stay_in_days"], axis=1)


#### train model ####
print(f"Predicting {target} ...")

# initialize lgb data structures
d_train = lgb.Dataset(train_features, label=train_labels, free_raw_data=True)
d_test = lgb.Dataset(
    test_features, label=test_labels, reference=d_train, free_raw_data=True
)
d_valid = lgb.Dataset(
    valid_features, label=valid_labels, reference=d_train, free_raw_data=True
)

# set training params
class_thresh = 0.5
params = config.C_READMIT_PARAMS_LGBM
gbm_model = lgb.LGBMClassifier(**params)

early_stopping_rounds = 200
if debug:
    early_stopping_rounds = 2
print(f"Early stopping rounds: {early_stopping_rounds}.")

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

gbm_model.fit(
    train_features,
    train_labels,
    eval_set=[(test_features, test_labels)],
    eval_metric="logloss",
    early_stopping_rounds=early_stopping_rounds,
)


#### evaluate model ####
evals_result = gbm_model._evals_result

metricsgen = lgbmClassificationHelpers(
    target,
    class_thresh,
    gbm_model,
    evals_result,
    features,
    labels,
    train_features,
    train_labels,
    test_features,
    test_labels,
    valid_features,
    valid_labels,
    figfolder,
    datafolder,
    modelfolder,
    tablefolder,
    calibrate_please=False,
)

metricsgen.lgbm_save_ttv_split()
pkl_model = metricsgen.lgbm_save_model_to_pkl_and_h5()

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")