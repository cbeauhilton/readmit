import sys

sys.path.append("modules")

from datetime import datetime


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


filename = config.PROCESSED_FINAL
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")


# readmitted30dpickle = data[configcols.READMITTED30_KEEP_COLS]
# readmitted03dpickle = data[configcols.READMITTED03_KEEP_COLS]
los7pickle = data[configcols.LOS7_KEEP_COLS]
losRpickle = data[configcols.LOSR_KEEP_COLS]
genderpickle = data[configcols.GENDER_KEEP_COLS]
insurancepickle = data[configcols.INSURANCE_KEEP_COLS]
racepickle = data[configcols.RACE_KEEP_COLS]

file_title = f"los7pickle_final_"
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".pickle"
title = file_title + timestr + ext
datafolder = config.PROCESSED_DATA_DIR / "targeted_pickles"
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
pkl_file = datafolder / title
with open(pkl_file, "wb") as fout:
    pickle.dump(los7pickle, fout)
print("Pickle file available at", pkl_file)

file_title = f"losRpickle_final_"
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".pickle"
title = file_title + timestr + ext
datafolder = config.PROCESSED_DATA_DIR / "targeted_pickles"
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
pkl_file = datafolder / title
with open(pkl_file, "wb") as fout:
    pickle.dump(losRpickle, fout)
print("Pickle file available at", pkl_file)

file_title = f"genderpickle_final_"
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".pickle"
title = file_title + timestr + ext
datafolder = config.PROCESSED_DATA_DIR / "targeted_pickles"
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
pkl_file = datafolder / title
with open(pkl_file, "wb") as fout:
    pickle.dump(genderpickle, fout)
print("Pickle file available at", pkl_file)


file_title = f"insurancepickle_final_"
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".pickle"
title = file_title + timestr + ext
datafolder = config.PROCESSED_DATA_DIR / "targeted_pickles"
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
pkl_file = datafolder / title
with open(pkl_file, "wb") as fout:
    pickle.dump(insurancepickle, fout)
print("Pickle file available at", pkl_file)

file_title = f"racepickle_final_"
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".pickle"
title = file_title + timestr + ext
datafolder = config.PROCESSED_DATA_DIR / "targeted_pickles"
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
pkl_file = datafolder / title
with open(pkl_file, "wb") as fout:
    pickle.dump(racepickle, fout)
print("Pickle file available at", pkl_file)

# file_title = f"readmitted30dpickle_final_"
# timestr = time.strftime("%Y-%m-%d-%H%M")
# timestrfolder = time.strftime("%Y-%m-%d")
# ext = ".pickle"
# title = file_title + timestr + ext
# datafolder = config.PROCESSED_DATA_DIR / "targeted_pickles"
# if not os.path.exists(datafolder):
#     print("Making folder called", datafolder)
#     os.makedirs(datafolder)
# pkl_file = datafolder / title
# with open(pkl_file, "wb") as fout:
#     pickle.dump(readmitted30dpickle, fout)
# print("Pickle file available at", pkl_file)

# file_title = f"readmitted03dpickle_final_"
# timestr = time.strftime("%Y-%m-%d-%H%M")
# timestrfolder = time.strftime("%Y-%m-%d")
# ext = ".pickle"
# title = file_title + timestr + ext
# datafolder = config.PROCESSED_DATA_DIR / "targeted_pickles"
# if not os.path.exists(datafolder):
#     print("Making folder called", datafolder)
#     os.makedirs(datafolder)
# pkl_file = datafolder / title
# with open(pkl_file, "wb") as fout:
#     pickle.dump(readmitted03dpickle, fout)
# print("Pickle file available at", pkl_file)


# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
