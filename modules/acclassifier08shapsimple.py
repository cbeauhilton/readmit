import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import UndefinedMetricWarning

import config
import configcols
from zz_generalHelpers import (
    lgb_f1_score,
    make_datafolder_for_target,
    make_figfolder_for_target,
    make_modelfolder_for_target,
    make_report_tables_folder,
    train_test_valid_80_10_10_split,
)
from zz_lgbmHelpers import lgbmClassificationHelpers
from zz_shapHelpers import shapHelpers

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

try:
    import cPickle as pickle
except BaseException:
    import pickle

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

# target = "age_gt_65_y"
# name_for_figs = "Age"

target = "length_of_stay_over_7_days"
name_for_figs = "Length of stay over 7 days"

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
#     "age_gt_65_y",
#     # "gender_binary",
#     # "race_binary",
# ]

shap_index = 500

# Load CSV of top SHAP values, and select the first n
csv_dir = config.SHAP_CSV_DIR
shap_file = f"{target}_shap.csv"
print("SHAP CSV:", csv_dir / shap_file)
top_shaps = pd.read_csv(csv_dir / shap_file)
top_shaps = top_shaps.rename(index=str, columns={"0": "feature_names"})
top_shaps = top_shaps[:shap_index]
shap_list = top_shaps["feature_names"].tolist()
shap_list.append(target)  # to make the labels and features sets
# print(shap_list)

filename = config.PROCESSED_FINAL
print("Loading", filename)
data = pd.read_pickle(filename)

print("File loaded.")

seed = config.SEED
debug = False
print("Debug:", debug)

figfolder = make_figfolder_for_target(debug, target)
datafolder = make_datafolder_for_target(debug, target)
modelfolder = make_modelfolder_for_target(debug, target)
tablefolder = make_report_tables_folder(debug)

data = data[shap_list]

train_set, test_set, valid_set = train_test_valid_80_10_10_split(
    data, target, seed
)

train_labels = train_set[target]
train_features = train_set.drop([target], axis=1)

test_labels = test_set[target]
test_features = test_set.drop([target], axis=1)

valid_labels = valid_set[target]
valid_features = valid_set.drop([target], axis=1)

labels = data[target]
features = data.drop([target], axis=1)

print("Predicting", target, "with", shap_index, "top features...")
d_train = lgb.Dataset(train_features, label=train_labels, free_raw_data=True)

d_test = lgb.Dataset(
    test_features, label=test_labels, reference=d_train, free_raw_data=True
)

d_valid = lgb.Dataset(
    valid_features, label=valid_labels, reference=d_train, free_raw_data=True
)

class_thresh = 0.65

params = config.C_READMIT_PARAMS_LGBM
early_stopping_rounds = 200
# gbm_model = lgb.LGBMClassifier(**params) # <-- could also do this, but it's kind of nice to have it all explicit
gbm_model = lgb.LGBMClassifier(
    boosting_type=params["boosting_type"],
    colsample_bytree=params["colsample_bytree"],
    is_unbalance=params["is_unbalance"],
    learning_rate=params["learning_rate"],
    max_depth=params["max_depth"],
    min_child_samples=params["min_child_samples"],
    min_child_weight=params["min_child_weight"],
    min_split_gain=params["min_split_gain"],
    n_estimators=params["n_estimators"],
    n_jobs=params["n_jobs"],
    num_leaves=params["num_leaves"],
    num_rounds=params["num_rounds"],
    objective=params["objective"],
    predict_contrib=params["predict_contrib"],
    random_state=params["random_state"],
    reg_alpha=params["reg_alpha"],
    reg_lambda=params["reg_lambda"],
    silent=params["silent"],
    subsample_for_bin=params["subsample_for_bin"],
    subsample_freq=params["subsample_freq"],
    subsample=params["subsample"],
)

gbm_model.fit(
    train_features,
    train_labels,
    eval_set=[(test_features, test_labels)],
    eval_metric="logloss",
    early_stopping_rounds=early_stopping_rounds,
)
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

metricsgen.lgbm_save_ttv_split() # make sure to save the ttv first
# the function deletes old h5 files with the same name
pkl_model = metricsgen.lgbm_save_model_to_pkl_and_h5() # this one uses "append" for the h5 part
metricsgen.lgbm_save_feature_importance_plot()
metricsgen.lgbm_classification_results()

with open(pkl_model, "rb") as fin:
    gbm_model = pickle.load(fin)
print(f"{target} Generating SHAP values...")
explainer = shap.TreeExplainer(gbm_model)
features_shap = features.sample(n=20000, random_state=seed, replace=False)
shap_values = explainer.shap_values(features_shap)

helpshap = shapHelpers(
    target, features_shap, shap_values, figfolder, datafolder, modelfolder
)
helpshap.shap_save_to_disk()
helpshap.shap_save_ordered_values()
helpshap.shap_prettify_column_names(
    prettycols_file=config.PRETTIFYING_COLUMNS_CSV
)
helpshap.shap_plot_summaries(
    title_in_figure=f"Impact of Variables on {name_for_figs} Prediction"
)
helpshap.shap_random_force_plots(
    n_plots=20, expected_value=explainer.expected_value
)
helpshap.shap_top_dependence_plots(n_plots=10)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")