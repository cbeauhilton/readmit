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

pd.options.display.max_columns = 2000


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikit's f1 doesn't like probabilities
    return "f1", f1_score(y_true, y_hat), True


def classifiermany(
    test_targets,
    debugfraction=0.005,
    class_thresh=0.20,
    debug=False,
    trainmodels=False,
    generateshap=False,
    generatemetrics=False,
    undersampling=False,
):
    test_targets = test_targets
    debugfraction = debugfraction

    if trainmodels:
        for target in test_targets:
            if undersampling:
                print("Undersampling...")
                under_sample_file = config.UNDERSAMPLE_TRAIN_SET
                test_set_file = config.TEST_SET
                valid_set_file = config.VALID_SET
                filename = config.PROCESSED_FINAL
                data_train = pd.read_pickle(under_sample_file)
                data_test = pd.read_pickle(test_set_file)
                data_valid = pd.read_pickle(valid_set_file)
                data = pd.read_pickle(filename)
            else:
                filename = config.PROCESSED_FINAL
                print("Loading files...")
                data = pd.read_pickle(filename)
                train_file = config.TRAIN_SET
                test_set_file = config.TEST_SET
                valid_set_file = config.VALID_SET
                data_train = pd.read_pickle(train_file)
                data_test = pd.read_pickle(test_set_file)
                data_valid = pd.read_pickle(valid_set_file)
                data = pd.read_pickle(filename)
                print(
                    "Files loaded:",
                    "\n",
                    filename,
                    "\n",
                    train_file,
                    "\n",
                    test_set_file,
                    "\n",
                    valid_set_file,
                )
            if debug:
                print(f"Selecting small portion ({debugfraction}) for debugging...")
                data_train = data_train.sample(
                    frac=debugfraction, random_state=config.SEED
                )
                data_test = data_test.sample(
                    frac=debugfraction, random_state=config.SEED
                )
                data_valid = data_valid.sample(
                    frac=debugfraction, random_state=config.SEED
                )
                data = data.sample(frac=debugfraction, random_state=config.SEED)
            else:
                pass

            if target in configcols.C_READMIT_BINARY_TARGETS:
                print(f"Removing deceased patients to predict {target}...")
                print("Length before dropping", len(data))
                data = data[data["dischargedispositiondescription"] != "Expired"]
                data_train = data_train[
                    data_train["dischargedispositiondescription"] != "Expired"
                ]
                data_test = data_test[
                    data_test["dischargedispositiondescription"] != "Expired"
                ]
                data_valid = data_valid[
                    data_valid["dischargedispositiondescription"] != "Expired"
                ]
                print("Length after dropping", len(data))
            elif target not in configcols.C_READMIT_BINARY_TARGETS:
                pass

            if target in configcols.SILLY_TARGETS:
                print(f"Dropping nulls for {target}...")
                print("Length before dropping:", len(data))
                data_train = data_train[data_train[target].notnull()]
                data_test = data_test[data_test[target].notnull()]
                data_valid = data_valid[data_valid[target].notnull()]
                data = data[data[target].notnull()]
                print("Length after dropping:", len(data))
            else:
                print("Data length:", len(data))

            c_train_labels = data_train[target]
            c_test_labels = data_test[target]
            c_valid_labels = data_valid[target]

            # Set threshold based on prevalence of positive target
            negativeclassnum = len(data[data[target] == 0])
            positiveclassnum = len(data[data[target] == 1])
            proportionposneg = positiveclassnum / negativeclassnum
            class_thresh = proportionposneg
            print(f"Threshold for {target} is {class_thresh}")

            if target in configcols.SILLY_TARGETS:
                class_thresh = 0.5
                print(f"Just kidding, threshold for {target} is {class_thresh}")

            c_train_features = data_train.drop(
                configcols.C_READMIT_DROP_COLS, axis=1, errors="ignore"
            )
            c_train_features = c_train_features.drop(
                configcols.DATETIME_COLS, axis=1, errors="ignore"
            )

            c_test_features = data_test.drop(
                configcols.C_READMIT_DROP_COLS, axis=1, errors="ignore"
            )
            c_test_features = c_test_features.drop(
                configcols.DATETIME_COLS, axis=1, errors="ignore"
            )

            c_valid_features = data_valid.drop(
                configcols.C_READMIT_DROP_COLS, axis=1, errors="ignore"
            )
            c_valid_features = c_valid_features.drop(
                configcols.DATETIME_COLS, axis=1, errors="ignore"
            )

            c_features = data.drop(
                configcols.C_READMIT_DROP_COLS, axis=1, errors="ignore"
            )
            c_features = c_features.drop(
                configcols.DATETIME_COLS, axis=1, errors="ignore"
            )

            if target == "financialclass_binary":
                print(f"Target is {target}, dropping raw insurance column...")
                c_train_features = c_train_features.drop(
                    "insurance2", axis=1, errors="ignore"
                )
                c_test_features = c_test_features.drop(
                    "insurance2", axis=1, errors="ignore"
                )
                c_valid_features = c_valid_features.drop(
                    "insurance2", axis=1, errors="ignore"
                )
                c_features = c_features.drop("insurance2", axis=1, errors="ignore")

            if target == "discharged_in_past_30d":
                c_train_features = c_train_features.drop(
                    [
                        "discharged_in_past_30d",
                        "days_between_current_admission_and_previous_discharge",
                    ],
                    axis=1,
                    errors="ignore",
                )
                c_test_features = c_test_features.drop(
                    [
                        "discharged_in_past_30d",
                        "days_between_current_admission_and_previous_discharge",
                    ],
                    axis=1,
                    errors="ignore",
                )
                c_valid_features = c_valid_features.drop(
                    [
                        "discharged_in_past_30d",
                        "days_between_current_admission_and_previous_discharge",
                    ],
                    axis=1,
                    errors="ignore",
                )
                c_features = c_features.drop(
                    [
                        "discharged_in_past_30d",
                        "days_between_current_admission_and_previous_discharge",
                    ],
                    axis=1,
                    errors="ignore",
                )
            elif target != "discharged_in_past_30d":
                pass

            # Save list of feature names to file prior to dropping specific features
            # (in the "past_30d" problem above, just dropped the targets)
            # Then for final modeling we can copy-pasta the feature list
            # and pass it in, instead of doing all this sequential drop business

            # First make a new data folder
            timestrfolder = time.strftime("%Y-%m-%d")
            if debug:
                datafolder = (
                    config.PROCESSED_DATA_DIR / timestrfolder / "debug" / target
                )
                if not os.path.exists(datafolder):
                    print("Making folder called", datafolder)
                    os.makedirs(datafolder)
            else:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / target
                if not os.path.exists(datafolder):
                    print("Making folder called", datafolder)
                    os.makedirs(datafolder)

            feature_list = list(c_train_features)
            df = pd.DataFrame(feature_list, columns=["features"])
            spreadsheet_title = f"Feature list 05 {target} "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".csv"
            title = spreadsheet_title + timestr + ext
            feature_list_file = datafolder / title
            df.to_csv(feature_list_file, index=False)

            if target in configcols.C_READMIT_BINARY_TARGETS:
                readmit_targets = configcols.C_READMIT_BINARY_TARGETS
                c_train_features = c_train_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                c_test_features = c_test_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                c_valid_features = c_valid_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                c_features = c_features.drop(readmit_targets, axis=1, errors="ignore")

            elif target in configcols.C_LOS_BINARY_TARGETS:
                # Drop Obs patients
                print("Dropping Obs patients...")
                print(len(data))
                data_train = data_train[
                    data_train["patientclassdescription"] != "Observation"
                ]
                data_test = data_test[
                    data_test["patientclassdescription"] != "Observation"
                ]
                data_valid = data_valid[
                    data_valid["patientclassdescription"] != "Observation"
                ]
                data = data[data["patientclassdescription"] != "Observation"]
                print(len(data))

                discharge_lab_cols = configcols.DISCHARGE_LAB_COLS
                discharge_other_cols = configcols.DISCHARGE_OTHER_COLS
                los_dependent = configcols.LOS_DEPENDENT
                los_targets = configcols.C_LOS_BINARY_TARGETS
                readmit_targets = configcols.C_READMIT_BINARY_TARGETS
                all_the_cols = [
                    # date_cols,
                    discharge_lab_cols,
                    discharge_other_cols,
                    # gen_drop_cols,
                    los_dependent,
                    los_targets,
                    readmit_targets,
                    # target,
                ]
                for col in all_the_cols:
                    print("Dropping", col)
                    c_train_features = c_train_features.drop(
                        col, axis=1, errors="ignore"
                    )
                    c_test_features = c_test_features.drop(col, axis=1, errors="ignore")

                    c_valid_features = c_valid_features.drop(
                        col, axis=1, errors="ignore"
                    )
                    c_features = c_features.drop(col, axis=1, errors="ignore")

            elif target == "gender_binary":
                gender_dependent = configcols.GENDER_DEPENDENT
                c_train_features = c_train_features.drop(
                    gender_dependent, axis=1, errors="ignore"
                )
                c_test_features = c_test_features.drop(
                    gender_dependent, axis=1, errors="ignore"
                )
                c_valid_features = c_valid_features.drop(
                    gender_dependent, axis=1, errors="ignore"
                )
                c_features = c_features.drop(gender_dependent, axis=1, errors="ignore")

            elif target == "race_binary":
                race_cols = configcols.RACE_COLS
                c_train_features = c_train_features.drop(
                    race_cols, axis=1, errors="ignore"
                )
                c_test_features = c_test_features.drop(
                    race_cols, axis=1, errors="ignore"
                )
                c_valid_features = c_valid_features.drop(
                    race_cols, axis=1, errors="ignore"
                )
                c_features = c_features.drop(race_cols, axis=1, errors="ignore")

            elif target == "financialclass_binary":
                financial_cols = configcols.FINANCIAL_COLS
                c_train_features = c_train_features.drop(
                    financial_cols, axis=1, errors="ignore"
                )
                c_test_features = c_test_features.drop(
                    financial_cols, axis=1, errors="ignore"
                )
                c_valid_features = c_valid_features.drop(
                    financial_cols, axis=1, errors="ignore"
                )
                c_features = c_features.drop(financial_cols, axis=1, errors="ignore")
                readmit_targets = configcols.C_READMIT_BINARY_TARGETS
                c_train_features = c_train_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                c_test_features = c_test_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                c_valid_features = c_valid_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                c_features = c_features.drop(readmit_targets, axis=1, errors="ignore")

            file_title = f"{target} c_test_features "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            ext = ".pickle"
            title = file_title + timestr + ext
            data_file = datafolder / title
            c_test_features.to_pickle(data_file)

            file_title = f"{target} c_test_labels "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                datafolder = (
                    config.PROCESSED_DATA_DIR / timestrfolder / "debug" / target
                )
            else:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / target
            data_file = datafolder / title
            c_test_labels.to_pickle(data_file)

            file_title = f"{target} c_features "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                datafolder = (
                    config.PROCESSED_DATA_DIR / timestrfolder / "debug" / target
                )
            else:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / target
            data_file = datafolder / title
            c_features.to_pickle(data_file)

            feature_list = list(c_train_features)
            df = pd.DataFrame(feature_list, columns=["features"])
            spreadsheet_title = f"Feature list 05 {target} "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".csv"
            title = spreadsheet_title + timestr + ext
            feature_list_file = datafolder / title
            df.to_csv(feature_list_file, index=False)

            print("Predicting", target)
            c_d_train = lgb.Dataset(
                c_train_features, label=c_train_labels, free_raw_data=True
            )
            c_d_test = lgb.Dataset(
                c_test_features,
                label=c_test_labels,
                reference=c_d_train,
                free_raw_data=True,
            )
            c_d_valid = lgb.Dataset(
                c_valid_features,
                label=c_valid_labels,
                reference=c_d_train,
                free_raw_data=True,
            )
            c_evals_result = {}  # to record eval results for plotting
            c_params = config.C_READMIT_PARAMS_LGBM
            early_stopping_rounds = 200
            if debug:
                early_stopping_rounds = 10
            c_gbm = lgb.train(
                c_params,
                c_d_train,
                num_boost_round=10_000_000,
                valid_sets=[c_d_test, c_d_train],
                valid_names=["test", "train"],
                feval=lgb_f1_score,
                evals_result=c_evals_result,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=25,
            )

            print("Dumping model with pickle...")
            file_title = f"{target} MODEL "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                modelfolder = config.MODELS_DIR / timestrfolder / "debug"
                if not os.path.exists(modelfolder):
                    print("Making folder called", modelfolder)
                    os.makedirs(modelfolder)
            else:
                modelfolder = config.MODELS_DIR / timestrfolder
                if not os.path.exists(modelfolder):
                    print("Making folder called", modelfolder)
                    os.makedirs(modelfolder)
            pkl_model = modelfolder / title
            with open(pkl_model, "wb") as fout:
                pickle.dump(c_gbm, fout)
            if generatemetrics:
                print("Generating results, tables, and figures...")
                # predict probabilities
                ax = lgb.plot_metric(c_evals_result, figsize=(5, 7), metric="f1")
                figure_title = f"{target} F1 Metric "
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title = figure_title + timestr + ext
                timestrfolder = time.strftime("%Y-%m-%d")
                if debug:
                    figfolder = config.FIGURES_DIR / timestrfolder / "debug" / target
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                else:
                    figfolder = config.FIGURES_DIR / timestrfolder / target
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                plt.savefig(
                    (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                )
                c_predict_labels = c_gbm.predict(c_test_features)
                probs = c_predict_labels

                fpr, tpr, threshold = metrics.roc_curve(c_test_labels, c_predict_labels)
                roc_auc = metrics.auc(fpr, tpr)

                print("Plot feature importances...")
                ax = lgb.plot_importance(
                    c_gbm, figsize=(5, 20), importance_type="gain", precision=2
                )
                figure_title = f"{target} Feature Importances AUC %0.2f_" % roc_auc
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title = figure_title + timestr + ext
                timestrfolder = time.strftime("%Y-%m-%d")
                if debug:
                    figfolder = config.FIGURES_DIR / timestrfolder / "debug" / target
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                else:
                    figfolder = config.FIGURES_DIR / timestrfolder / target
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                plt.savefig(
                    (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                )
                # plt.show()
                plt.close()

                print("Generating ROC curve...")
                # plt.figure(figsize=(5,5))
                plt.title("Receiver Operating Characteristic")
                plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
                plt.legend(loc="lower right")
                plt.plot([0, 1], [0, 1], "r--")
                plt.xlim([-0.011, 1.011])
                plt.ylim([-0.011, 1.011])
                plt.ylabel("True Positive Rate")
                plt.xlabel("False Positive Rate")
                figure_title = (
                    f"{target} Receiver_Operating_Characteristic_AUC_%0.2f_" % roc_auc
                )
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title = figure_title + timestr + ext
                plt.savefig(
                    (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                )
                # plt.show()
                plt.close()
                print(f"{target} ROC AUC %0.2f" % roc_auc)

                print("Generating PR curve...")
                average_precision = average_precision_score(
                    c_test_labels, c_predict_labels
                )

                precision, recall, _ = precision_recall_curve(
                    c_test_labels, c_predict_labels
                )

                step_kwargs = (
                    {"step": "post"}
                    if "step" in signature(plt.fill_between).parameters
                    else {}
                )
                plt.title(
                    " {0} Precision-Recall Curve AP {1:0.2f}".format(
                        target, average_precision
                    )
                )
                plt.step(recall, precision, color="b", alpha=0.2, where="post")
                plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                figure_title = (
                    f"{target} Precision-Recall curve AP=%0.2f " % average_precision
                )
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title1 = figure_title + timestr + ext
                print(title1)
                plt.savefig(
                    (figfolder / title1),
                    dpi=400,
                    transparent=False,
                    bbox_inches="tight",
                )
                plt.close()
                print(
                    f"{target} Average precision-recall score: {0:0.2f}".format(
                        average_precision
                    )
                )

                print("Generating calibration curve...")
                brier_score = brier_score_loss(c_test_labels, c_predict_labels)
                print("Brier score without optimized calibration: %1.3f" % brier_score)
                gb_y, gb_x = calibration_curve(
                    c_test_labels, c_predict_labels, n_bins=50
                )
                plt.plot([0, 1], [0, 1], linestyle="--")
                # plot model reliability
                plt.plot(gb_x, gb_y, marker=".")
                plt.suptitle(f"Calibration plot for {target}")
                plt.xlabel("Predicted probability")
                plt.ylabel("True probability")
                figure_title = f"{target} Calibration curve {brier_score} "
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title1 = figure_title + timestr + ext
                print(title1)
                plt.savefig(
                    (figfolder / title1),
                    dpi=400,
                    transparent=False,
                    bbox_inches="tight",
                )
                plt.close()

                print("Generating classification report...")
                c_predict_labels = c_gbm.predict(c_test_features)
                c_range = c_predict_labels.size

                # convert into binary values for classification report
                print("Classification threshold is ", class_thresh)
                for i in range(0, c_range):
                    if (
                        c_predict_labels[i] >= class_thresh
                    ):  # set threshold to desired value
                        c_predict_labels[i] = 1
                    else:
                        c_predict_labels[i] = 0

                accuracy = metrics.accuracy_score(c_test_labels, c_predict_labels)
                print(f"Accuracy of GBM classifier for {target}: ", accuracy)
                print(classification_report_imbalanced(c_test_labels, c_predict_labels))

                # report = classification_report_imbalanced(c_test_labels, c_predict_labels)
                # df = pd.DataFrame(report).transpose()
                # df.to_csv("clf_report.csv")

                conf_mx = metrics.confusion_matrix(c_test_labels, c_predict_labels)
                fig = sns.heatmap(conf_mx, square=True, annot=True, fmt="d", cbar=False)
                fig = fig.get_figure()
                plt.xlabel("True Label")
                plt.ylabel("Predicted Label")
                figure_title = f"{target} Confusion Matrix AUC %0.2f_" % roc_auc
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title = figure_title + timestr + ext
                plt.savefig(
                    (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                )
                plt.close()

                conf_mx = pd.DataFrame(conf_mx)

                FP = conf_mx.sum(axis=0) - np.diag(conf_mx)
                FN = conf_mx.sum(axis=1) - np.diag(conf_mx)
                TP = np.diag(conf_mx)
                TN = conf_mx.values.sum() - (FP + FN + TP)

                # Sensitivity, hit rate, recall, or true positive rate
                TPR = TP / (TP + FN)
                # Specificity or true negative rate
                TNR = TN / (TN + FP)
                # Precision or positive predictive value
                PPV = TP / (TP + FP)
                # Negative predictive value
                NPV = TN / (TN + FN)
                # Fall out or false positive rate
                FPR = FP / (FP + TN)
                # False negative rate
                FNR = FN / (TP + FN)
                # False discovery rate
                FDR = FP / (TP + FP)

                # Overall accuracy
                ACC = (TP + TN) / (TP + FP + FN + TN)

                print(target)
                n_pts = len(c_predict_labels)
                print("N:", n_pts)
                print("TPR:", TPR)
                print("TNR:", TNR)
                print("PPV:", PPV)
                print("NPV:", NPV)
                print("FPR:", FPR)
                print("FNR:", FNR)
                print("FDR:", FDR)
                print("ACC:", ACC)

                d = [
                    [
                        timestr,
                        target,
                        class_thresh,
                        n_pts,
                        TPR,
                        TNR,
                        PPV,
                        NPV,
                        FPR,
                        FNR,
                        FDR,
                        ACC,
                        roc_auc,
                        average_precision,
                        brier_score,
                    ]
                ]

                df = pd.DataFrame(
                    d,
                    columns=(
                        "Time",
                        "Target",
                        "Threshold",
                        "Number of Patients",
                        "TPR",
                        "TNR",
                        "PPV",
                        "NPV",
                        "FPR",
                        "FNR",
                        "FDR",
                        "ACC",
                        "ROC AUC",
                        "Average Precision",
                        "Brier Score Loss",
                    ),
                )
                if debug:
                    df.to_csv("debugreports.csv", mode="a", header=True)
                else:
                    df.to_csv(config.TRAINING_REPORTS, mode="a", header=True)

                # Find 95% Confidence Interval
                auc, auc_cov = z_compare_auc_delong_xu.delong_roc_variance(
                    c_test_labels, c_predict_labels
                )

                auc_std = np.sqrt(auc_cov)
                alpha = 0.95
                lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

                ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

                ci[ci > 1] = 1

                print(f"{target} AUC:", auc)
                print("AUC COV:", auc_cov)
                print("95% AUC CI:", ci)
            if generateshap:
                with open(pkl_model, "rb") as fin:
                    c_gbm = pickle.load(fin)
                    
                print(f"{target} Generating SHAP values...")
                startTime1 = datetime.now()
                print(startTime1)
                explainer = shap.TreeExplainer(c_gbm)
                c_features_shap = c_features.sample(
                    n=20000, random_state=seed, replace=False
                )
                shap_values = explainer.shap_values(c_features_shap)
                print(f"{target} Saving to disk...")
                file_title = f"{target} SHAP "
                timestr = time.strftime("%Y-%m-%d-%H%M")
                timestrfolder = time.strftime("%Y-%m-%d")
                ext = ".pickle"
                title = file_title + timestr + ext
                if debug:
                    modelfolder = config.MODELS_DIR / timestrfolder / "debug"
                    if not os.path.exists(modelfolder):
                        print("Making folder called", modelfolder)
                        os.makedirs(modelfolder)
                else:
                    modelfolder = config.MODELS_DIR / timestrfolder
                    if not os.path.exists(modelfolder):
                        print("Making folder called", modelfolder)
                        os.makedirs(modelfolder)
                full_path = modelfolder / title
                # Pickled numpy array
                with open(full_path, "wb") as f:
                    pickle.dump(shap_values, f)
                # Use the prettified column names CSV for showing results
                # First copy the SHAP features,
                # standardize the column names to the pattern I used
                # to make the prettycol csv,
                # then load the pretty cols,
                # make a dict out of the old and new columns
                # and map the new onto the old
                result = c_features_shap.copy()
                result.columns = (
                    result.columns.str.strip()
                    .str.replace("\t", "")
                    .str.replace("_", " ")
                    .str.replace("__", " ")
                    .str.replace(", ", " ")
                    .str.replace(",", " ")
                    .str.replace("'", "")
                    .str.capitalize()
                )
                prettycols_file = config.PRETTIFYING_COLUMNS_CSV
                prettycols = pd.read_csv(prettycols_file)
                di = prettycols.set_index("feature_ugly").to_dict()
                result.columns = result.columns.to_series().map(di["feature_pretty"])

                shap.summary_plot(
                    shap_values,
                    c_features_shap,
                    title="Impact of Variables on Readmission Prediction",
                    plot_type="bar",
                    show=False,
                    feature_names=result.columns,
                )
                figure_title = f"{target} SHAP_summary_bar "
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title = figure_title + timestr + ext
                if debug:
                    figfolder = config.FIGURES_DIR / timestrfolder / "debug" / target
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                else:
                    figfolder = config.FIGURES_DIR / timestrfolder / target
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                plt.savefig(
                    (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                )
                plt.close()
                print(f"{target} Making SHAP summary plot...")
                shap.summary_plot(
                    shap_values,
                    c_features_shap,
                    title="Impact of Variables on Readmission Prediction",
                    feature_names=result.columns,
                    show=False,
                )
                figure_title = f"{target}_SHAP_summary "
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title = figure_title + timestr + ext
                plt.savefig(
                    (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                )
                plt.close()

                shap_indices = np.random.choice(
                    shap_values.shape[0], 5
                )  # select 5 random patients
                for pt_num in shap_indices:
                    print(f"{target} Making force plot for patient", pt_num, "...")
                    shap.force_plot(
                        explainer.expected_value,  # this version uses the standard base value
                        shap_values[pt_num, :],
                        c_features_shap.iloc[
                            pt_num, :
                        ],  # grabs the identities and values of components
                        text_rotation=30,  # easier to read
                        matplotlib=True,  # instead of Javascript
                        show=False,  # allows saving, etc.
                        link="logit",
                        feature_names=result.columns,
                    )
                    figure_title = f"{target}_SHAP_"
                    patient_number = f"_Pt_{pt_num}"
                    timestr = time.strftime("%Y-%m-%d-%H%M")
                    ext = ".png"
                    title = figure_title + timestr + patient_number + ext
                    forcefolder = figfolder / "force_plots"
                    if not os.path.exists(forcefolder):
                        print("Making folder called", forcefolder)
                        os.makedirs(forcefolder)
                    plt.savefig(
                        (forcefolder / title),
                        dpi=400,
                        transparent=False,
                        bbox_inches="tight",
                    )
                    plt.close()

                # Dependence plots don't like the "category" dtype, will only work with
                # category codes (ints).
                # Make a copy of the df that uses cat codes
                # Then use the original df as the "display"
                df_with_codes = c_features_shap.copy()
                df_with_codes.columns = (
                    df_with_codes.columns.str.strip()
                    .str.replace("_", " ")
                    .str.replace("__", " ")
                    .str.capitalize()
                )
                for col in list(df_with_codes.select_dtypes(include="category")):
                    df_with_codes[col] = df_with_codes[col].cat.codes

                c_features_shap.columns = (
                    c_features_shap.columns.str.strip()
                    .str.replace("\t", "")
                    .str.replace("_", " ")
                    .str.replace("__", " ")
                    .str.replace(", ", " ")
                    .str.replace(",", " ")
                    .str.replace("'", "")
                    .str.capitalize()
                )
                prettycols_file = config.PRETTIFYING_COLUMNS_CSV
                prettycols = pd.read_csv(prettycols_file)
                di = prettycols.set_index("feature_ugly").to_dict()
                c_features_shap.columns = c_features_shap.columns.to_series().map(di["feature_pretty"])

                print("Saving SHAP values to disk in order of importance...")
                df_shap_train = pd.DataFrame(
                    shap_values, columns=c_features_shap.columns.values
                )
                imp_cols = (
                    df_shap_train.abs()
                    .mean()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                pickle_title = f"{target}_shap_df.pickle"
                df_shap_train.to_pickle(datafolder / pickle_title)
                imp_cols = pd.DataFrame(imp_cols)
                print(imp_cols)
                csv_title = f"{target}_shap.csv"
                imp_cols.to_csv(datafolder / csv_title)

                c_features_shap = c_features_shap.copy()
                for i in range(10):
                    shap.dependence_plot(
                        "rank(%d)" % i,
                        shap_values,
                        df_with_codes,
                        display_features=c_features_shap,
                        show=False,
                        feature_names=c_features_shap.columns,
                    )
                    print(f"Making dependence plot for {target} feature ranked {i}...")
                    figure_title = f"{target}_SHAP_dependence_{i}_"
                    timestr = time.strftime("%Y-%m-%d-%H%M")
                    ext = ".png"
                    title = figure_title + timestr + ext
                    plt.savefig(
                        (figfolder / title),
                        dpi=400,
                        transparent=False,
                        bbox_inches="tight",
                    )
                    plt.close()


if __name__ == "__main__":
    classifiermany()


##########################################################################

# data_train1 = data_train.sample(frac=0.5, random_state=seed, replace=False)
# data_test = data_train.loc[set(data_train.index) - set(data_train1.index)]
# data_train = data_train1 # make the test set 50:50 readmission:no readmission


# data_train = data[
#     (data["admissiontime"] > config.TRAIN_START)
#     & (data["admissiontime"] < config.TRAIN_END)
# ]

# data_test = data[
#     (data["admissiontime"] > config.TEST_START)
#     & (data["admissiontime"] < config.TEST_END)
# ]

# data_valid = data[
#     (data["admissiontime"] > config.VALID_START)
#     & (data["admissiontime"] < config.VALID_END)
# ]


# for pt_num in shap_indices: # this version includes base value as a feature
#     print("Making force plot for patient ", pt_num, "...")
#     shap.force_plot(
#         0,  # set expected value to zero
#         np.hstack(
#             [shap_values[pt_num, :], explainer.expected_value]
#         ),  # add expected value as a bar in the force plot
#         pd.concat(
#             [
#                 c_features_shap,
#                 pd.DataFrame(
#                     {"Base value": [explainer.expected_value]}
#                 ),
#             ],
#             axis=1,
#         ).iloc[
#             pt_num, :
#         ],  # grabs the identities and values of components
#         text_rotation=30,  # easier to read
#         matplotlib=True,  # instead of Javascript
#         show=False,  # allows saving, etc.
#         link="logit"
#     )


##########################################################################
