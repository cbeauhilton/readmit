import os
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd

import config
import configcols

from sklearn.exceptions import UndefinedMetricWarning

from zz_generalHelpers import (
    lgb_f1_score,
    make_datafolder_for_target,
    make_figfolder_for_target,
    make_modelfolder_for_target,
    make_report_tables_folder,
    train_test_valid_80_10_10_split,
)
from zz_lgbmHelpers import lgbmRegressionHelpers
from zz_shapHelpers import shapHelpers

try:
    import cPickle as pickle
except BaseException:
    print("No cPickle!")
    import pickle

print("About to run", os.path.basename(__file__))
startTime = datetime.now()


filename = config.PROCESSED_FINAL
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")

target = "length_of_stay_in_days"
# target = "days_between_current_discharge_and_next_admission"
# target = "patient_age"

# data = data[configcols.LOSR_KEEP_COLS]
seed = config.SEED


# name_for_figs = "Length of Stay"
name_for_figs = "Readmission"

debug = False

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


print("Predicting", target)
d_train = lgb.Dataset(train_features, label=train_labels, free_raw_data=True)

d_test = lgb.Dataset(
    test_features, label=test_labels, reference=d_train, free_raw_data=True
)

d_valid = lgb.Dataset(
    valid_features, label=valid_labels, reference=d_train, free_raw_data=True
)

evals_result = {}  # to record eval results for plotting
params = config.R_READMIT_PARAMS_LGBM
early_stopping_rounds = 200
gbm_model = lgb.train(
    params,
    d_train,
    num_boost_round=10_000_000,
    valid_sets=[d_test, d_train],
    valid_names=["test", "train"],
    early_stopping_rounds=early_stopping_rounds,
    evals_result=evals_result,
    verbose_eval=25,
)


metricsgen = lgbmRegressionHelpers(
    target,
    gbm_model,
    evals_result,
    test_features,
    test_labels,
    figfolder,
    datafolder,
    modelfolder,
    tablefolder,
)

pkl_model = metricsgen.lgbm_save_model_to_pickle()
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
# helpshap.shap_save_ordered_values()
helpshap.shap_prettify_column_names(prettycols_file=config.PRETTIFYING_COLUMNS_CSV)
helpshap.shap_plot_summaries(
    title_in_figure=f"Impact of Variables on {name_for_figs} Prediction"
)
helpshap.shap_random_force_plots(n_plots=5, expected_value=explainer.expected_value)
helpshap.shap_top_dependence_plots(n_plots=10)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
