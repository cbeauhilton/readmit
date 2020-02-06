#%%
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
import h5py
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import glob

jsonpickle_numpy.register_handlers()
import json
import traceback

import pandas as pd
import shap

from sklearn.externals import joblib
# print(os.getcwd())
# modules_dir = Path(os.getcwd()) / "modules"
# os.chdir(modules_dir)
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
# from cbh.lgbmHelpers import lgbmClassificationHelpers
from cbh.shapHelpers import shapHelpers

try:
    import cPickle as pickle
except BaseException:
    import pickle

# print("About to run", os.path.basename(__file__))
startTime = datetime.now()

seed = config.SEED
debug = False
# print("Debug:", debug)

"""
find the h5 file you are interested in at models\<datestr>\<target>\*.h5
"""
print("Loading path to h5...")
model_dir = config.MODELS_DIR
datestr = "2019-05-16"
target = "length_of_stay_over_3_days"
dirpath = model_dir / datestr / target
filenamez = [str(pp) for pp in dirpath.glob("*.h5")]
# assuming there's only one .h5 file here, but could select with the index at filenamez[x]
filename = Path(filenamez[0])
f = h5py.File(Path(filename), "r")
keylist = list(f.keys())

print("This h5 file contains", keylist)
keylist.remove("gbm_model")  # will be loaded separately

print("\n Loading all keys from .h5...")
for k in keylist:
    # this will make a new variable for every key in the .h5 file
    # and load the corresponding dataframes.
    # if you don't want everything loaded
    # use "keylist.remove("xyz")
    print(k)
    exec(f'{k} = pd.read_hdf(Path(r"{filename}"), key="{k}")')

# convert to numpy array, as required by shap
shap_values = shap_values.to_numpy(copy=True)
target = shap_expected_value.iloc[0]["target"]
name_for_figs = shap_expected_value.iloc[0]["name_for_figs"]
class_thresh = shap_expected_value.iloc[0]["class_thresh"]
shap_expected_val = shap_expected_value.iloc[0]["shap_exp_val"]

print(test_labels)
print("Target:", target)
print("Name for figs:", name_for_figs)
print("SHAP expected value:", shap_expected_val)
print("class_thresh:", class_thresh)

print("Loading model...")
h5_model_key = "gbm_model"
gbm_model = load_jsonified_sklearn_model_from_h5(filename, h5_model_key)

print("Files loaded.")

figfolder = make_figfolder_for_target(debug, target)
datafolder = make_datafolder_for_target(debug, target)
modelfolder = make_modelfolder_for_target(debug, target)
tablefolder = make_report_tables_folder(debug)
print("\n", figfolder, "\n", datafolder, "\n", modelfolder, "\n", tablefolder)

evals_result = gbm_model._evals_result

# metricsgen = lgbmClassificationHelpers(
#     target,
#     class_thresh,
#     gbm_model,
#     evals_result,
#     features,
#     labels,
#     train_features,
#     train_labels,
#     test_features,
#     test_labels,
#     valid_features,
#     valid_labels,
#     figfolder,
#     datafolder,
#     modelfolder,
#     tablefolder,
#     calibrate_please=False,
# )

# metricsgen.lgbm_classification_results()

helpshap = shapHelpers(
    target,
    name_for_figs,
    class_thresh,
    features_shap,
    shap_values,
    shap_expected_val,
    gbm_model,
    figfolder,
    datafolder,
    modelfolder,
)

helpshap.shap_prettify_column_names(prettycols_file=config.PRETTIFYING_COLUMNS_CSV)

# helpshap.shap_plot_summaries(
#     title_in_figure=f"Impact of Variables on {name_for_figs} Prediction"
# )
#%%
shap.initjs()
helpshap.shap_random_force_plots(n_plots=20, expected_value=shap_expected_val)
#%%
# helpshap.shap_top_dependence_plots(n_plots=10)

# helpshap.shap_top_dependence_plots_self(n_plots=20)

# helpshap.shap_int_vals_heatmap()

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")


###########################################################################################################################


# shap_expected_val = shap_expected_value.iloc[0][1]
# target = "readmitted30d"
# name_for_figs = "Readmission"
# class_thresh = 0.2

# helpshap.shap_save_ordered_values()

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

# print("Loading path to h5...")
# filename = Path(
#     r"C:\Users\hiltonc\Desktop\readmit\readmit\models\2019-05-14\readmitted30d\readmitted30d_284_everything__2019-05-14.h5"
# )
# f = h5py.File(filename, "r")
# keylist = list(f.keys())
# print("This h5 file contains", keylist)

# filename = Path(
#     r"C:\Users\hiltonc\Desktop\readmit\readmit\models\2019-05-14\length_of_stay_over_5_days\length_of_stay_over_5_days_172_everything__2019-05-14.h5"
# )

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

"""
Select your target and threshold from the list
"""
# target = "length_of_stay_over_5_days"
# name_for_figs = "Length of stay over 5 days"
# class_thresh = 0.2

# target = "age_gt_65_y"
# name_for_figs = "Age"
# class_thresh = 0.5

# target = "readmitted30d"
# name_for_figs = "Readmission"
# class_thresh = 0.2

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
###########################################################################################################################


#################################=================================#################################


# print("Loading files...")
# features = pd.read_hdf(filename, key="features")
# # print(features.head())
# # print(features['patientclassdescription'].value_counts(dropna=False))

# print("now the next one...")
# labels = pd.read_hdf(filename, key="labels")
# target = labels.name
# # print(target)

# print("and the next one...")
# train_features = pd.read_hdf(filename, key="train_features")
# # print(train_features.head())

# print("and the next one...")
# train_labels = pd.read_hdf(filename, key="train_labels")
# # print(train_labels.head())

# print("and the next one...")
# test_features = pd.read_hdf(filename, key="test_features")
# # print(test_features.head())

# print("and the next one...")
# test_labels = pd.read_hdf(filename, key="test_labels")
# # print(test_labels.head())

# print("and the next one...")
# valid_features = pd.read_hdf(filename, key="valid_features")
# # print(valid_features.head())

# print("and the next one...")
# valid_labels = pd.read_hdf(filename, key="valid_labels")
# # print(valid_labels.head())

# print("and the next one...")
# features_shap = pd.read_hdf(filename, key="features_shap")
# # print(features_shap.head())

# print("and the next one...")
# shap_vals = pd.read_hdf(filename, key="shap_values")
# convert to numpy array, as required by shap
# shap_values = shap_values.to_numpy(copy=True)
# print(shap_values[:10]) # no "head()" for np arrays

# print("and the next one...")
# requirements = pd.read_hdf(filename, key="requirements")
# # print(requirements.head())

# print("and the next one...")
# df_shap_train = pd.read_hdf(filename, key="df_shap_train")
# # print(df_shap_train.head())

# print("and the next one...")
# ordered_shap_cols = pd.read_hdf(filename, key="ordered_shap_cols")
# # print(ordered_shap_cols.head())

# print("and finally...")
# shap_expected_val  = pd.read_hdf(filename, key="shap_expected_value")
# target = shap_expected_val.iloc[0]["target"]
# name_for_figs = shap_expected_val.iloc[0]["name_for_figs"]
# class_thresh = shap_expected_val.iloc[0]["class_thresh"]
# shap_expected_val = shap_expected_value.iloc[0]["shap_exp_val"]


#################################=================================#################################
