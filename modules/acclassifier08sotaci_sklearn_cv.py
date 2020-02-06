import glob
import json
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import mlxtend
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from numpy import mean, median, percentile
from scipy import interp
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

from cbh import config
from cbh.generalHelpers import (get_latest_folders, load_ttv_split,
                                make_datafolder_for_target,
                                make_figfolder_for_target,
                                make_modelfolder_for_target,
                                make_report_tables_folder,
                                train_test_valid_80_10_10_split)
from cbh.lgbmHelpers import bootstrap_estimate_and_ci
from cbh.plottingHelpers import save_pr_curve, save_roc_auc

# try:
#     from sklearn.ensemble import HistGradientBoostingClassifier
# except:
#     from sklearn.experimental import enable_hist_gradient_boosting
#     from sklearn.ensemble import HistGradientBoostingClassifier
# try:
#     from sklearn.impute import IterativeImputer
# except:
#     from sklearn.experimental import enable_iterative_imputer
#     from sklearn.impute import IterativeImputer


# import lightgbm as lgb
# from fastai.callbacks import *
# from fastai.tabular import *

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

##################################################################################################################
target = config.TARGET
targets = config.LOS_TARGETS

n_splits = 5 
n_repeats = 1

for target in targets:
    figfolder = config.FIGURES_DIR/f"{target}"

    try:
        os.makedirs(figfolder)
    except FileExistsError:
        pass

    print(f"\n {target} file loading...")
    final_file = config.PROCESSED_DATA_DIR / f"{target}.h5"
    data = pd.read_hdf(final_file, key=f"{target}clean")

    data = data[:100_000]

    print("File loaded.")

    y = data[target]
    X = data.drop([target], axis=1)

    print("Dropping length of stay in days for LoS targets...")
    X = X.drop(["length_of_stay_in_days"], axis=1)


    # Set up calibrationFig. S1a curve
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy='median')), ("scaler", MinMaxScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

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

    # initialize scores dictionary
    clf_scores = {}

    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Complement Naive Bayes", ComplementNB()),
        ("Support Vector Machine", LinearSVC()),
    ]

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=config.SEED
    )


    # for ROC_AUC curve
    cls_fig_dict = {}

    for cls_name, classifier in classifiers:
        print(f"\n {cls_name} training...")


        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

        cls_fig_dict[cls_name] = {}
        cls_fig_dict[cls_name]["prob_pos"] = {}
        cls_fig_dict[cls_name]["y_test"] = {}
        cls_fig_dict[cls_name]["y_pred"] = {}
        scores = {}

        for score, _ in scores_list:
            scores[score] = []

        for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            print(f"Training loop {i+1}/{n_splits*n_repeats} for {target}...")

            pipe.fit(X_train, y_train)


            if hasattr(classifier, "predict_proba"):
                prob_pos = pipe.predict_proba(X_test)[:, 1]
            else:  # use decision function
                prob_pos = pipe.decision_function(X_test)
                prob_pos = \
                    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            
            y_pred = pipe.predict(X_test)

            cls_fig_dict[cls_name]["y_test"][i] = [y_test.tolist()]
            cls_fig_dict[cls_name]["prob_pos"][i] = [prob_pos.tolist()]
            cls_fig_dict[cls_name]["y_pred"][i] = [y_pred.tolist()]

            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, prob_pos, n_bins=10
            )

            ax1.plot(
                mean_predicted_value, fraction_of_positives, "s-", label=f"{cls_name}"
            )

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=cls_name, histtype="step", lw=2)

            for score, scorer in scores_list:
                try:
                    scores[score].append(scorer(y_test, prob_pos))
                except:
                    scores[score].append(scorer(y_test, y_pred))

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

            df_dict = {}
            df_dict["target"] = target
            df_dict["Classifier"] = cls_name

            for score, _ in scores_list:
                df_dict[score] = summary_scores[score]["summ"]

            df = pd.DataFrame([df_dict])
        
            df.set_index('target', inplace=True)

            print(df.head(5))
            df.to_csv(f"{cls_name}_{target}_summary_scores.csv")

            clf_scores[cls_name] = summary_scores 

            print(clf_scores)
        

        with open(f"{figfolder}/{target}_{cls_name.replace(' ', '_')}_scores.json", "w") as f:
            json.dump(scores, f, indent=4)




    print(clf_scores)

    with open(f"{figfolder}/scores.json", "w") as f:
        json.dump(clf_scores, f, indent=4)

    with open(f"{figfolder}/{target}_fig_dict.json", "w") as f:
        json.dump(cls_fig_dict, f, indent=4)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig(figfolder / f"{target}_sota_calibration.pdf")
    plt.close()

        # ("Gradient Boosting Machine", HistGradientBoostingClassifier(
        #         learning_rate=0.1,
        #         max_iter=10_000_000,
        #         max_leaf_nodes=99,
        #         min_samples_leaf=360,
        #         l2_regularization=0.149874,
        #         max_bins=63,
        #         warm_start=False,
        #         scoring='loss',
        #         validation_fraction=0.1,
        #         n_iter_no_change=200,
        #         tol=1e-07,
        #         verbose=1,
        #         random_state=config.SEED,
        #     )), #train loss: 0.54731, val loss: 0.59599
