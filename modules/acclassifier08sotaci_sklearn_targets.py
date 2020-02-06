import glob
import pickle
import time
import warnings

import json
import matplotlib.pyplot as plt
import mlxtend
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from sklearn.experimental import (enable_hist_gradient_boosting,
                                  enable_iterative_imputer)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
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

# import lightgbm as lgb
# from fastai.callbacks import *
# from fastai.tabular import *

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

##################################################################################################################
targets = config.LOS_TARGETS

for target in targets:

    print(f"==="*20)
    print(f"\n{target}\n")
    print(f"==="*20)

    #### Load data proper ####
    filename = config.PROCESSED_FINAL
    print("Loading", filename)
    data = pd.read_pickle(filename)

    print("File loaded.")

    debug = False
    print("Debug:", debug)

    if debug:
        data = data[:20000]

    # Load CSV of top SHAP values, and select the first n
    csv_dir = config.SHAP_CSV_DIR
    shap_file = f"{target}_shap.csv"
    # print("SHAP CSV:", csv_dir / shap_file)
    top_shaps = pd.read_csv(csv_dir / shap_file)
    top_shaps = top_shaps.rename(index=str, columns={"0": "feature_names"})
    shap_index = 22  # [10, 20, 30, 40, 50, 60, 500]
    top_shaps = top_shaps[:shap_index]
    shap_list = top_shaps["feature_names"].tolist()
    shap_list.append(target)  # to make the labels and features sets
    dont_misses = [
        "platelet_count_admit_value",
        "pressureulcer_Present_on_Admission_to_the_Hospital",
        "length_of_stay_in_days",
    ]
    for dont_miss in dont_misses:
        shap_list.append(dont_miss)
    # print(shap_list)

    # final cleaning for LoS prediction
    print("Dropping expired, obs, outpt, ambulatory, emergency patients...")
    data = data[data["dischargedispositiondescription"] != "Expired"]
    data = data[data["patientclassdescription"] != "Observation"]  #
    data = data[data["patientclassdescription"] != "Outpatient"]  # ~10,000
    data = data[
        data["patientclassdescription"] != "Ambulatory Surgical Procedures"
    ]  # ~8,000
    data = data[data["patientclassdescription"] != "Emergency"]  # ~7,000

    #### drop everything but the good stuff ####
    dropem = list(set(list(data)) - set(shap_list))
    data = data.drop(columns=dropem)

    final_file = config.PROCESSED_DATA_DIR / f"{target}.h5"
    print(f"Saving to {final_file}...")
    data.to_hdf(final_file, key=f"{target}clean", mode='a', format='table')

    sample_size = 500_000
    print(f"Saving data set with {sample_size} encounters for comparison with Rajkomar...")
    small_data = data.sample(n=sample_size)
    small_data.to_hdf(final_file, key=f"{target}cleansmall", mode='a', format='table')

    figfolder = make_figfolder_for_target(debug, target)
    datafolder = make_datafolder_for_target(debug, target)
    modelfolder = make_modelfolder_for_target(debug, target)
    tablefolder = make_report_tables_folder(debug)

    train_set, test_set, valid_set = train_test_valid_80_10_10_split(small_data, target, config.SEED)

    train_labels = train_set[target]
    train_features = train_set.drop([target], axis=1)

    test_labels = test_set[target]
    test_features = test_set.drop([target], axis=1)

    valid_labels = valid_set[target]
    valid_features = valid_set.drop([target], axis=1)

    labels = small_data[target]
    features = small_data.drop([target], axis=1)


    print("Dropping length of stay in days for LoS targets...")
    features = features.drop(["length_of_stay_in_days"], axis=1)
    train_features = train_features.drop(["length_of_stay_in_days"], axis=1)
    test_features = test_features.drop(["length_of_stay_in_days"], axis=1)
    valid_features = valid_features.drop(["length_of_stay_in_days"], axis=1)

    X_train = train_features
    y_train = train_labels
    X_test = test_features
    y_test = test_labels
    X = features
    y = labels


    # grab desired scoring functions
    scoring_funcs_0 = [average_precision_score, roc_auc_score]
    scoring_funcs_1 = [
        brier_score_loss,
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        matthews_corrcoef,
    ]

    n_splits = 5  # number of splits for bootstrap
    method = ".632"  # type of bootstrap evaluation method

    subsample_n = len(X)
    random_state = config.SEED

    idx = np.random.choice(np.arange(len(X)), subsample_n, replace=False)
    X_boot = X.iloc[idx]
    y_boot = y.iloc[idx]

    # Set up calibrationFig. S1a curve
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # initialize scores dictionary
    scores = {}

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy='median')), ("scaler", MinMaxScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_features = features.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = features.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Complement Naive Bayes", ComplementNB()),
        ("Support Vector Machine", LinearSVC()),
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
    ]


    for cls_name, classifier in classifiers:
        print(f"\n {cls_name} training...")

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        
        pipe.fit(X_train, y_train)


        if hasattr(classifier, "predict_proba"):
            prob_pos = pipe.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = pipe.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        save_pr_curve(
            target=target,
            classifier=cls_name,
            test_labels=y_test,
            predicted_labels=prob_pos,
            figfolder=figfolder,
        )

        save_roc_auc(
            target=target,
            classifier=cls_name,
            test_labels=y_test,
            predicted_labels=prob_pos,
            figfolder=figfolder,
        )

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10
        )

        ax1.plot(
            mean_predicted_value, fraction_of_positives, "s-", label=f"{cls_name}"
        )

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=cls_name, histtype="step", lw=2)

        ci_dict = {}
        ci_dict_1 = bootstrap_estimate_and_ci(
            pipe,
            # X_boot,
            X,
            # y_boot,
            y,
            scoring_func=scoring_funcs_1,
            n_splits=n_splits,
            method=method,
        )

        cloned_estimator = clone(classifier)
        if hasattr(classifier, "predict_proba"):
            cloned_estimator.predict = cloned_estimator.predict_proba
        else:  # use decision function
            cloned_estimator.predict = cloned_estimator.decision_function

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", cloned_estimator)])

        ci_dict_0 = bootstrap_estimate_and_ci(
            pipe,
            # X_boot,
            X,
            # y_boot,
            y,
            scoring_func=scoring_funcs_0,
            n_splits=n_splits,
            method=method,
        )

        ci_dict.update(ci_dict_0)
        ci_dict.update(ci_dict_1)

        scores[cls_name] = ci_dict
        print(scores)

    print(scores)

    with open(f"{config.TABLES_DIR}/{target}_scores.json", "w") as f:
        json.dump(scores, f, indent=4)


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
