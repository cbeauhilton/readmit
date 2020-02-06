import os
import glob
import time
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

import cbh.config as config
import cbh.configcols as configcols
from cbh.generalHelpers import (
    train_test_valid_80_10_10_split,
    get_latest_folders,
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
class_thresh = 0.5


target = config.TARGET 
name_for_figs = config.NAME_FOR_FIGS

# target = "readmitted30d"
# name_for_figs = "Readmission"

debug = False
print("Debug:", debug)

modelfolder, datafolder, figfolder, tablefolder = get_latest_folders(target)

def lgbm_load_ttv_split():
    h5_file = max(glob.iglob(str(modelfolder) + '/*.h5'), key=os.path.getmtime)
    print(f"Loading TTV split from {h5_file}...")

    train_features = pd.read_hdf(h5_file, key='train_features')
    train_labels = pd.read_hdf(h5_file, key='train_labels')
    test_features = pd.read_hdf(h5_file, key='test_features')
    test_labels = pd.read_hdf(h5_file, key='test_labels')
    valid_features = pd.read_hdf(h5_file, key='valid_features')
    valid_labels = pd.read_hdf(h5_file, key='valid_labels')
    labels = pd.read_hdf(h5_file, key='labels')
    features = pd.read_hdf(h5_file, key='features')

    return train_features, train_labels, test_features, test_labels, valid_features, valid_labels, labels, features

train_features, train_labels, test_features, test_labels, valid_features, valid_labels, labels, features = lgbm_load_ttv_split()


# find most recently modified pickle file in the model folder
pkl_model = max(glob.iglob(os.path.join(modelfolder, '*MODEL*.pickle')), key=os.path.getmtime)

with open(pkl_model, "rb") as f:
    print(f"Loading model from {pkl_model}...")
    gbm_model = pickle.load(f)

evals_result = None

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

metricsgen.lgbm_save_feature_importance_plot()
metricsgen.lgbm_classification_results()

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")