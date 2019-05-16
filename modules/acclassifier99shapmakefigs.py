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

from sklearn.externals import joblib

import cbh.config as config
import configcols
from cbh.generalHelpers import (
    lgb_f1_score,
    make_datafolder_for_target,
    make_figfolder_for_target,
    make_modelfolder_for_target,
    make_report_tables_folder,
    train_test_valid_80_10_10_split,
    load_jsonified_sklearn_model_from_h5,
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
debug = False
print("Debug:", debug)


"""
Select your target and threshold from the list
"""
# target = "length_of_stay_over_5_days"
# name_for_figs = "Length of stay over 5 days"
# class_thresh = 0.2

# target = "age_gt_65_y"
# name_for_figs = "Age"
# class_thresh = 0.5

target = "readmitted30d"
name_for_figs = "Readmission"
class_thresh = 0.2

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

"""
find the h5 file you are interested in at models\<date>\<target>\*.h5
"""

# print("Loading path to h5...")
# filename = Path(
#     r"C:\Users\hiltonc\Desktop\readmit\readmit\models\2019-05-14\readmitted30d\readmitted30d_284_everything__2019-05-14.h5"
# )
# f = h5py.File(filename, "r")
# keylist = list(f.keys())
# print("This h5 file contains", keylist)

print("Loading path to h5...")
filename = Path(
    r"C:\Users\hiltonc\Desktop\readmit\readmit\models\2019-05-14\length_of_stay_over_5_days\length_of_stay_over_5_days_172_everything__2019-05-14.h5"
)
f = h5py.File(filename, "r")
keylist = list(f.keys())
print("This h5 file contains", keylist)
#################################=================================#################################

print("Loading model...")
h5_model_key = "gbm_model"
gbm_model = load_jsonified_sklearn_model_from_h5(filename, h5_model_key)

print("Loading first file...")
features = pd.read_hdf(filename, key="features")
# print(features.head())
print(features['patientclassdescription'].value_counts(drop))

print("now the next one...")
labels = pd.read_hdf(filename, key="labels")
# print(labels.head())

print("and the next one...")
train_features = pd.read_hdf(filename, key="train_features")
# print(train_features.head())

print("and the next one...")
train_labels = pd.read_hdf(filename, key="train_labels")
# print(train_labels.head())

print("and the next one...")
test_features = pd.read_hdf(filename, key="test_features")
# print(test_features.head())

print("and the next one...")
test_labels = pd.read_hdf(filename, key="test_labels")
# print(test_labels.head())

print("and the next one...")
valid_features = pd.read_hdf(filename, key="valid_features")
# print(valid_features.head())

print("and the next one...")
valid_labels = pd.read_hdf(filename, key="valid_labels")
# print(valid_labels.head())

print("and the next one...")
features_shap = pd.read_hdf(filename, key="features_shap")
# print(features_shap.head())

print("and the next one...")
shap_vals = pd.read_hdf(filename, key="shap_values")
# convert to numpy array, as required by shap
shap_values = shap_vals.to_numpy(copy=True)
# print(shap_values[:10]) # no "head()" for np arrays

print("and the next one...")
requirements = pd.read_hdf(filename, key="requirements")
# print(requirements.head())

print("and the next one...")
df_shap_train = pd.read_hdf(filename, key="df_shap_train")
# print(df_shap_train.head())

print("and the next one...")
ordered_shap_cols = pd.read_hdf(filename, key="ordered_shap_cols")
# print(ordered_shap_cols.head())

print("and finally...")
shap_expected_val  = pd.read_hdf(filename, key="shap_expected_value")
# expected value is in the first row, second column:
shap_expected_value = shap_expected_val.iloc[0][1]
# print(shap_expected_value)

print("Files loaded.")


#################################=================================#################################


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

helpshap = shapHelpers(target, features_shap, shap_values, shap_expected_value, gbm_model, figfolder, datafolder, modelfolder)

helpshap.shap_save_ordered_values()

helpshap.shap_prettify_column_names(prettycols_file=config.PRETTIFYING_COLUMNS_CSV)

print("Loading path to h5...")
filenamex = Path(
    r"C:\Users\hiltonc\Desktop\readmit\readmit\models\2019-05-15\readmitted30d\readmitted30d_284_everything__2019-05-15_.h5"
)
fx = h5py.File(filenamex, "r")
keylistx = list(fx.keys())
print("This h5 file contains", keylistx)
pretty_ordered_shap_cols = pd.read_hdf(filenamex, key="pretty_shap_cols")
# print(pretty_ordered_shap_cols.head(111))
pretty_imp_cols = pd.read_hdf(filenamex, key="pretty_imp_cols")
print(pretty_imp_cols.head(111))

helpshap.shap_plot_summaries(
    title_in_figure=f"Impact of Variables on {name_for_figs} Prediction"
)
helpshap.shap_random_force_plots(n_plots=20, expected_value=shap_expected_value)

helpshap.shap_top_dependence_plots(n_plots=10)

helpshap.shap_top_dependence_plots_self(n_plots=20)

helpshap.shap_int_vals_heatmap()

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")




###########################################################################################################################

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


# filename1 = Path(
#     r"C:\Users\hiltonc\Desktop\readmit\readmit\models\2019-05-14\readmitted30d\readmitted30d_SHAP_values_284_2019-05-14-1554.pickle"
# )
# print("Loading", filename1)
# shap_vals_pkl = pd.read_pickle(filename1)
# print("File loaded.")
# print(shap_vals_pkl.head())

# print("Loading path to h5...")
# filenamex = Path(
#     r"C:\Users\hiltonc\Desktop\readmit\readmit\models\2019-05-15\readmitted30d\readmitted30d_284_everything__2019-05-15_.h5"
# )
# fx = h5py.File(filenamex, "r")
# keylistx = list(fx.keys())
# print("This h5 file contains", keylistx)
# pretty_ordered_shap_cols = pd.read_hdf(filenamex, key="pretty_ordered_shap_cols")
# print(pretty_ordered_shap_cols.head())


###########################################################################################################################