import ast
import csv
import os
import sys
import time
import warnings
from datetime import datetime
from timeit import default_timer as timer

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import shap
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.stochastic import sample
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.fixes import signature


sys.path.append("modules")
import config
import z_compare_auc_delong_xu

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)


MAX_EVALS = 500
N_FOLDS = 10


try:
    import cPickle as pickle
except BaseException:
    import pickle


seed = config.SEED

pd.options.display.max_columns = 2000

gender_dependent = ["attending_specialty_institute_desc"]

los_dependent = [
    "infectiousdiseaseconsult",
    "pt_ot_consult",
    "opiatesduringadmit",
    "benzosduringadmit",
    "lineinfection",
    "cdiffinfection",
    "fallduringadmission",
    "spiritualcareconsult",
]

discharge_info = [
    "abs_baso_discharge_value",
    "abs_eosin_discharge_value",
    "abs_lymph_discharge_value",
    "abs_mono_discharge_value",
    "abs_neut_anc_discharge_value",
    "absolute_nrbc_discharge_value",
    "albumin_discharge_value",
    "alkaline_phosphatase_discharge_value",
    "alt_discharge_value",
    "ast_discharge_value",
    "baso_discharge_value",
    "bilirubin_total_discharge_value",
    "bmi_discharge",
    "bun_discharge_value",
    "calcium_discharge_value",
    "chloride_discharge_value",
    "co2_discharge_value",
    "creatinine_discharge_value",
    "diff_type_discharge_value",
    "discharge_day_of_week",
    "discharge_diastolic_bp",
    "discharge_hour_of_day",
    "discharge_systolic_bp",
    "discharged_on_holiday",
    "discharged_on_weekend",
    "dischargedispositiondescription",
    "dischargedonbenzo",
    "dischargedonopiate",
    "dischargemeds",
    "eosin_discharge_value",
    "heartrate_discharge",
    "hematocrit_discharge_value",
    "hemoglobin_discharge_value",
    "length_of_stay_in_days",
    "lymph_discharge_value",
    "mch_discharge_value",
    "mchc_discharge_value",
    "mcv_discharge_value",
    "medsondischargedate",
    "mono_discharge_value",
    "mpv_discharge_value",
    "neut_discharge_value",
    "nucleated_reds_discharge_value",
    "platelet_count_discharge_value",
    "potassium_discharge_value",
    "rbc_discharge_value",
    "rdw_discharge_value",
    "sodium_discharge_value",
    "temperature_discharge",
    "wbc_discharge_value",
]


test_targets = [
    "readmitted30d",
    "died_within_48_72h_of_admission_combined",
    "length_of_Stay_over_5_days",
    "length_of_Stay_over_7_days",
    "length_of_Stay_over_14_days",
    "readmitted0.5d",
    "readmitted15d",
    "readmitted180d",
    "readmitted1d",
    "readmitted28d",
    "readmitted3650d",
    "readmitted365d",
    "readmitted3d",
    "readmitted45d",
    "readmitted7d",
    "readmitted90d",
    # "readmittedpast30d",
    # "gender",
]

filename = config.TRAIN_TEST_SET
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")

# Convert to numpy array for splitting in cross validation
labels = data.readmitted30d
features = data.drop(test_targets, axis=1)
features = features.drop(config.C_READMIT_DROP_COLS_LGBM_MANY, axis=1)
features = features.drop("dischargedispositiondescription", axis=1)
# test_features = np.array(test)

# Create a lgb dataset
train_set = lgb.Dataset(features, label=labels)

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])

    start = timer()

    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 10000, nfold = n_folds,
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50) #lgb_f1_score

    run_time = timer() - start

    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])

    # Loss must be minimized
    loss = 1 - best_score

    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'train_time': run_time, 'status': STATUS_OK}


# Define the search space
space = {
    'is_unbalance': hp.choice('is_unbalance', ['true', 'false']),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    # "verbose": -1, #-1 only shows errors
}


# optimization algorithm
tpe_algorithm = tpe.suggest


# Keep track of results
bayes_trials = Trials()

# File to save first results
out_file = 'gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()


# Global variable
global  ITERATION

ITERATION = 0

# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(seed))


print("Loading hyperopt results...")
results = pd.read_csv("gbm_trials.csv")

# Sort with best scores on top and reset index for slicing
results.sort_values("loss", ascending=True, inplace=True)
results.reset_index(inplace=True, drop=True)
print(results.head())


# Convert from a string to a dictionary
ast.literal_eval(results.loc[0, "params"])


# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, "estimators"])
best_bayes_params = ast.literal_eval(results.loc[0, "params"]).copy()
print("Best params...")
print(best_bayes_params)


print("Re-create the best model and train on the training data...")
best_bayes_model = lgb.LGBMClassifier(
    n_estimators=best_bayes_estimators,
    n_jobs=-1,
    objective="binary",
    random_state=seed,
    **best_bayes_params
)
print(best_bayes_model)

best_bayes_model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    class_weight=None,
    colsample_bytree=0.707630032256903,
    importance_type="split",
    is_unbalance="false",
    learning_rate=0.010302298912236304,
    max_depth=-1,
    min_child_samples=360,
    min_child_weight=0.001,
    min_split_gain=0.0,
    n_estimators=568,
    n_jobs=-1,
    num_leaves=99,
    objective="binary",
    random_state=42,
    reg_alpha=0.5926734167821595,
    reg_lambda=0.1498749826768534,
    silent=False,
    subsample=0.6027609913849075,
    subsample_for_bin=240000,
    subsample_freq=0,
)

best_bayes_model.fit(features, labels)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")

