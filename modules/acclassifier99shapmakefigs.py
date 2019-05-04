import os
import time
import warnings
from datetime import datetime
from pathlib import Path
import h5py
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()
import json
import traceback

# import lightgbm as lgb
# import numpy as np
import pandas as pd
import shap

# from sklearn.exceptions import UndefinedMetricWarning
from sklearn.externals import joblib

import config
import configcols
from zz_generalHelpers import (
    lgb_f1_score,
    make_datafolder_for_target,
    make_figfolder_for_target,
    make_modelfolder_for_target,
    make_report_tables_folder,
    train_test_valid_80_10_10_split,
    load_jsonified_sklearn_model_from_h5,
)
from zz_lgbmHelpers import lgbmClassificationHelpers
from zz_shapHelpers import shapHelpers

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
# warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

try:
    import cPickle as pickle
except BaseException:
    import pickle

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

# target = "length_of_stay_over_5_days"
# name_for_figs = "Length of stay over 5 days"

# target = "age_gt_65_y"
# name_for_figs = "Age"
target = "readmitted30d"
name_for_figs = "Readmission"

seed = config.SEED
debug = False
print("Debug:", debug)
class_thresh = 0.65

# target = [
#     # "readmitted30d",
#     # "readmitted5d",
#     # "readmitted7d",
#     # "readmitted3d",
#     # "length_of_stay_over_3_days",
#     # "length_of_stay_over_5_days",
#     # "length_of_stay_over_7_days",
#     # "financialclass_binary",
#     # "age_gt_10_y",
#     # "age_gt_30_y",
#     # "age_gt_65_y",
#     # "gender_binary",
#     # "race_binary",
# ]

shap_index = 500

print("Loading files...")
filename = Path(
    r"C:\Users\hiltonc\Desktop\readmit\models\2019-04-09\readmitted30d\readmitted30d_285_everything__2019-04-09.h5"
)
h5_model_key = "gbm_model"
f = h5py.File(filename, "r")
keylist = list(f.keys())
print("This h5 file contains", keylist)


# def load_jsonified_sklearn_model_from_h5(filename, h5_model_key):
#     with h5py.File(filename, "r") as f:
#         print("Loading JSON model as clf...")
#         h5_json_load = json.loads(
#             f[h5_model_key][()]
#         )  # takes a string and returns a dictionary
#         h5_json_model = json.dumps(
#             h5_json_load
#         )  # takes a dictionary and returns a string
#         clf = jsonpickle.decode(h5_json_model)  # requires a string, not dict
#         print(clf)
#         return clf


gbm_model = load_jsonified_sklearn_model_from_h5(filename, h5_model_key)

# with h5py.File(filename, 'r') as f:
#     print("Loading JSON gbm_model")
#     h5_json_load = json.loads(f['gbm_model'][()]) # takes a string and returns a dictionary
#     h5_json_model = json.dumps(h5_json_load) # takes a dictionary and returns a string
#     clf = jsonpickle.decode(h5_json_model) # requires a string, not dict
#     print(clf)

# gbm_model = clf
# print("Loading pickled gbm_model...")
# gbm_model_file = Path(r"C:\Users\hiltonc\Desktop\readmit\models\2019-04-09\readmitted30d\readmitted30d_285_features_MODEL_285_2019-04-09-1116_.pickle")
# with open(gbm_model_file, 'rb') as f:
#     gbm_model = pickle.load(f)
#     print(gbm_model)

print("Loading first file...")
train_features = pd.read_hdf(filename, key="train_features")
print("now the next one...")
train_labels = pd.read_hdf(filename, key="train_labels")
print("now the next one...")
test_features = pd.read_hdf(filename, key="test_features")
print("now the next one...")
test_labels = pd.read_hdf(filename, key="test_labels")
print("now the next one...")
valid_features = pd.read_hdf(filename, key="valid_features")
print("now the next one...")
valid_labels = pd.read_hdf(filename, key="valid_labels")
print("now the next one...")
features = pd.read_hdf(filename, key="features")
print("now the next one...")
labels = pd.read_hdf(filename, key="labels")
print("now the next one...")
features_shap = pd.read_hdf(filename, key="features_shap")
print("now the next one...")
shap_values = pd.read_hdf(filename, key="shap_values")
print("Files loaded.")


figfolder = make_figfolder_for_target(debug, target)
datafolder = make_datafolder_for_target(debug, target)
modelfolder = make_modelfolder_for_target(debug, target)
tablefolder = make_report_tables_folder(debug)
print(figfolder, datafolder, modelfolder, tablefolder)

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

# metricsgen.lgbm_save_ttv_split()
# metricsgen.lgbm_save_feature_importance_plot()
# metricsgen.lgbm_classification_results()

# print(f"{target} Generating SHAP values...")
explainer = shap.TreeExplainer(gbm_model)
features_shap = features.sample(n=20000, random_state=seed, replace=False)
shap_values = explainer.shap_values(features_shap)
print(explainer.expected_value)

helpshap = shapHelpers(
    target, features_shap, shap_values, gbm_model, figfolder, datafolder, modelfolder
)
# helpshap.shap_save_to_disk()
# helpshap.shap_save_ordered_values()
helpshap.shap_prettify_column_names(prettycols_file=config.PRETTIFYING_COLUMNS_CSV)
helpshap.shap_plot_summaries(
    title_in_figure=f"Impact of Variables on {name_for_figs} Prediction"
)
helpshap.shap_random_force_plots(n_plots=20, expected_value=explainer.expected_value)
helpshap.shap_top_dependence_plots(n_plots=10)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
