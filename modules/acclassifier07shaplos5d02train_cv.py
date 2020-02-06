import os
import time
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean, median, percentile
import json

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

import cbh.config as config
import cbh.configcols as configcols

try:
    import cPickle as pickle
except BaseException:
    import pickle


print("About to run", os.path.basename(__file__))
startTime = datetime.now()
seed = config.SEED

target = config.TARGET
targets = config.LOS_TARGETS

early_stopping_rounds = 200
n_splits = 10
n_repeats = 10

for target in targets:
    final_file = config.PROCESSED_DATA_DIR / f"{target}.h5"
    data = pd.read_hdf(final_file, key=f"{target}clean")

    # data = data[:100_000]

    print("File loaded.")

    y = data[target]
    X = data.drop([target], axis=1)

    print("Dropping length of stay in days for LoS targets...")
    X = X.drop(["length_of_stay_in_days"], axis=1)

    #### train model ####
    print(f"Predicting {target} ...")

    # set training params
    params = config.C_READMIT_PARAMS_LGBM
    clf = lgb.LGBMClassifier(**params)

    print(f"Early stopping rounds: {early_stopping_rounds}.")

    # When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
    warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

    scores_list = [
        ("ROC AUC", roc_auc_score),
        ("Average Precision", average_precision_score),
        ("Precision", precision_score),
        ("Recall", recall_score),
        ("Accuracy", accuracy_score),
        ("F1 Score", f1_score),
        ("Matthews Correlation Coefficient", matthews_corrcoef),
        ("Brier Score Loss", brier_score_loss),
    ]

    scores = {}

    for score, _ in scores_list:
        scores[score] = []

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=config.SEED
    )

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds,
        )

        prob_pos = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        for score, scorer in scores_list:
            try:
                scores[score].append(scorer(y_test, prob_pos))
            except:
                scores[score].append(scorer(y_test, y_pred))

        print(scores)

    summary_scores = {}
    for score, _ in scores_list:
        score_mean = mean(scores[score])
        score_median = median(scores[score])
        score_2_5th = percentile(scores[score], 2.5)
        score_97_th = percentile(scores[score], 97.5)

        summary_scores[score] = {
            "mean": score_mean, 
            "median": score_median,
            "2.5th": score_2_5th,
            "97.5th": score_97_th,
            "summ": f"{score_mean:.3f} [{score_2_5th:.3f}-{score_97_th:.3f}]"
            }

    print(summary_scores)

    with open(f"{target}_summary_scores.json", "w") as f:
        json.dump(summary_scores, f, indent=4)

    df_dict = {}
    df_dict["target"] = target
    for score, _ in scores_list:
        df_dict[score] = summary_scores[score]["summ"]

    df = pd.DataFrame([df_dict])
    print(df.head(5))

    df.to_csv(f"{target}_summary_scores.csv")

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
