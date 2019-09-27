import sys

sys.path.append("modules")


from datetime import datetime
import time

startTime = datetime.now()

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import numpy as np
from imblearn.metrics import classification_report_imbalanced
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

try:
    import cPickle as pickle
except BaseException:
    import pickle

import config

seed = config.SEED

print("Loading model from pickle...")

pkl_model = config.LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE
# load model with pickle to predict
with open(pkl_model, "rb") as fin:
    c_gbm = pickle.load(fin)

c_features = pd.read_pickle(config.C_FEATURES_FILE)

print("Generating SHAP values...")
startTime1 = datetime.now()
print(startTime1)
explainer = shap.TreeExplainer(c_gbm)
shap_values = explainer.shap_values(c_features)

print("Saving to disk...")

# Raw numpy array
np.save(config.C_LGBM_SHAP_FILE_NP, shap_values)
# Pickled numpy array
with open(config.C_LGBM_SHAP_FILE, 'wb') as f:
    pickle.dump(shap_values, f)

# shap_file = config.C_LGBM_SHAP_FILE
# shap_values.to_pickle(shap_file)

# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
