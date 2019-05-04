import sys

sys.path.append("modules")

from datetime import datetime

from pathlib import Path
import os
import pandas as pd
import lightgbm as lgb
import time
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import z_compare_auc_delong_xu
import scipy.stats
from scipy import stats
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import RandomUnderSampler
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.utils.fixes import signature

try:
    import cPickle as pickle
except BaseException:
    import pickle

import config
import configcols


print("About to run", os.path.basename(__file__))
startTime = datetime.now()

seed = config.SEED

file_name = Path(r"C:\Users\hiltonc\Desktop\readmit\data\processed\targeted_pickles\los7pickle_final_2019-03-19-1703.pickle")
data = pd.read_pickle(file_name)
c_train_labels = data["length_of_stay_over_7_days"]
c_train_features = data.drop(["length_of_stay_over_7_days"], axis=1)

c_d_train = lgb.Dataset(
    c_train_features, label=c_train_labels, free_raw_data=True
)

c_evals_result = {}  # to record eval results for plotting
c_params = config.C_READMIT_PARAMS_LGBM
early_stopping_rounds = 200
c_gbm = lgb.cv(
    c_params,
    c_d_train,
    num_boost_round=10000000,
    metrics = "auc",
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=25,
    nfold=10,
    stratified=True, #default is true
    seed=seed,
    show_stdv = True,
)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
