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
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.utils.fixes import signature

try:
    import cPickle as pickle
except BaseException:
    import pickle

import config

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

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


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return "f1", f1_score(y_true, y_hat), True


def classifiermany(
    test_targets,
    debugfraction=0.005,
    class_thresh=0.20,
    debug=False,
    trainmodels=False,
    generateshap=False,
    generatemetrics=False,
):
    test_targets = test_targets
    debugfraction = debugfraction
    filename = config.PROCESSED_FINAL
    print("Loading files...")
    data = pd.read_pickle(filename)
    train_test_file = config.TRAIN_TEST_SET
    valid_set_file = config.VALID_SET
    data_train_test = pd.read_pickle(train_test_file)
    data_valid = pd.read_pickle(valid_set_file)
    data = pd.read_pickle(filename)
    print("Files loaded.")
    if trainmodels:
        if debug:
            print(f"Selecting small portion ({debugfraction}) for debugging...")
            data_train_test = data_train_test.sample(frac=debugfraction, random_state=config.SEED)
            data_valid = data_valid.sample(frac=debugfraction, random_state=config.SEED)
            data = data.sample(frac=debugfraction, random_state=config.SEED)
        else:
            pass
        data_train_test = data_train_test.drop(config.C_READMIT_DROP_COLS_LGBM_MANY, axis=1)
        data_valid = data_valid.drop(config.C_READMIT_DROP_COLS_LGBM_MANY, axis=1)
        data = data.drop(config.C_READMIT_DROP_COLS_LGBM_MANY, axis=1)
        for target in test_targets:
            c_train_labels = data_train_test[target]
            c_train_features = data_train_test.drop(test_targets, axis=1)
            c_valid_labels = data_valid[target]
            c_valid_features = data_valid.drop(test_targets, axis=1)
            c_features = data.drop(test_targets, axis=1)
            if (
                target == "Length_of_Stay_over_5_days"
                or "Length_of_Stay_over_7_days"
                or "Length_of_Stay_over_14_days"
            ):
                c_train_features = c_train_features.drop(los_dependent, axis=1)
                c_valid_features = c_valid_features.drop(los_dependent, axis=1)
                c_features = c_features.drop(los_dependent, axis=1)
                c_train_features = c_train_features.drop(discharge_info, axis=1)
                c_valid_features = c_valid_features.drop(discharge_info, axis=1)
                c_features = c_features.drop(discharge_info, axis=1)
            elif target == "died_within_48_72h_of_admission_combined":
                c_train_features = c_train_features[data.length_of_stay_in_days > 1]
                c_valid_features = c_valid_features[data.length_of_stay_in_days > 1]
                c_features = c_features[data.length_of_stay_in_days > 1]
                c_train_features = c_train_features[data.length_of_stay_in_days < 3]
                c_valid_features = c_valid_features[data.length_of_stay_in_days < 3]
                c_features = c_features[data.length_of_stay_in_days < 3]
                c_train_features = c_train_features.drop(discharge_info, axis=1)
                c_valid_features = c_valid_features.drop(discharge_info, axis=1)
                c_features = c_features.drop(discharge_info, axis=1)
            elif target == "gender":
                c_train_features = c_train_features.drop(discharge_info, axis=1)
                c_valid_features = c_valid_features.drop(discharge_info, axis=1)
                c_features = c_features.drop(discharge_info, axis=1)
            file_title = f"{target} c_train_features "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / "debug"
                if not os.path.exists(datafolder):
                    print("Making folder called", datafolder)
                    os.makedirs(datafolder)
            else:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder
                if not os.path.exists(datafolder):
                    print("Making folder called", datafolder)
                    os.makedirs(datafolder)
            data_file = datafolder / title
            c_train_features.to_pickle(data_file)

            file_title = f"{target} c_train_labels "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / "debug"
            else:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder
            data_file = datafolder / title
            c_train_labels.to_pickle(data_file)

            file_title = f"{target} c_features "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / "debug"
            else:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder
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
            c_gbm = lgb.cv(
                c_params,
                c_d_train,
                num_boost_round=10000000,
                # feval=lgb_f1_score,
                metrics = "auc",
                # evals_result=c_evals_result,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=25,
                nfold=10,
                stratified=True #default is true
            )
            # Highest score
            c_gbm_best = np.max(c_gbm['auc-mean'])
            # Standard deviation of best score
            c_gbm_best_std = c_gbm['auc-stdv'][np.argmax(c_gbm['auc-mean'])]
            print('The maximium ROC AUC on the validation set was {:.5f} with std of {:.5f}.'.format(c_gbm_best, c_gbm_best_std))
            print('The ideal number of iterations was {}.'.format(np.argmax(c_gbm['auc-mean']) + 1))
            # early_stopping_rounds = 100
            # if debug:
            #     early_stopping_rounds = 10
            # c_gbm = lgb.train(
            #     c_params,
            #     c_d_train,
            #     num_boost_round=100_000,
            #     valid_sets=c_d_test,
            #     early_stopping_rounds=early_stopping_rounds,
            #     verbose_eval=25,
            #     evals_result=c_evals_result,
            #     keep_training_booster=False,
            # )

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
                print("Generating results, tables, and figures...") ###TODO: Tweak this for CV
                # predict probabilities
                ax = lgb.plot_metric(c_evals_result, figsize=(5, 7), metric="f1")
                figure_title = f"{target} F1 Metric "
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title = figure_title + timestr + ext
                timestrfolder = time.strftime("%Y-%m-%d")
                if debug:
                    figfolder = config.FIGURES_DIR / timestrfolder / "debug"
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                else:
                    figfolder = config.FIGURES_DIR / timestrfolder
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
                    figfolder = config.FIGURES_DIR / timestrfolder / "debug"
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                else:
                    figfolder = config.FIGURES_DIR / timestrfolder
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

                print("Generating PR curve...")

                average_precision = average_precision_score(
                    c_test_labels, c_predict_labels
                )

                print(
                    "Average precision-recall score: {0:0.2f}".format(average_precision)
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
                    " {0} Precision-Recall Curve AP {1:0.2f}".format(target, average_precision)
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
                    (figfolder / title1), dpi=400, transparent=False, bbox_inches="tight"
                )
                # plt.show(block=False)
                plt.close()

                print("Generating classification report...")
                c_predict_labels = c_gbm.predict(c_test_features)
                c_range = c_predict_labels.size

                # convert into binary values for classification report
                for i in range(0, c_range):
                    if (
                        c_predict_labels[i] >= class_thresh
                    ):  # set threshold to desired value
                        c_predict_labels[i] = 1
                    else:
                        c_predict_labels[i] = 0

                accuracy = metrics.accuracy_score(c_test_labels, c_predict_labels)
                print("Accuracy of GBM classifier: ", accuracy)
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

                print("TPR:", TPR)
                print("TNR:", TNR)
                print("PPV:", PPV)
                print("NPV:", NPV)
                print("FPR:", FPR)
                print("FNR:", FNR)
                print("FDR:", FDR)
                print("ACC:", ACC)

                # Find 95% Confidence Interval
                auc, auc_cov = z_compare_auc_delong_xu.delong_roc_variance(
                    c_test_labels, c_predict_labels
                )

                auc_std = np.sqrt(auc_cov)
                alpha = 0.95
                lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

                ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

                ci[ci > 1] = 1

                print("AUC:", auc)
                print("AUC COV:", auc_cov)
                print("95% AUC CI:", ci)
            if generateshap:
                with open(pkl_model, "rb") as fin:
                    c_gbm = pickle.load(fin)
                print("Generating SHAP values...")
                startTime1 = datetime.now()
                print(startTime1)
                explainer = shap.TreeExplainer(c_gbm)
                c_features_shap = c_features.sample(n=20000, random_state=seed, replace=False)
                shap_values = explainer.shap_values(c_features_shap)
                print("Saving to disk...")
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
                shap.summary_plot(
                    shap_values,
                    c_features_shap,
                    title="Impact of Variables on Readmission Prediction",
                    plot_type="bar",
                    show=False,
                )
                figure_title = f"{target} SHAP_summary_bar "
                timestr = time.strftime("%Y-%m-%d-%H%M")
                ext = ".png"
                title = figure_title + timestr + ext
                if debug:
                    figfolder = config.FIGURES_DIR / timestrfolder / "debug"
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                else:
                    figfolder = config.FIGURES_DIR / timestrfolder
                    if not os.path.exists(figfolder):
                        print("Making folder called", figfolder)
                        os.makedirs(figfolder)
                plt.savefig(
                    (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                )
                plt.close()
                print("Making SHAP summary plot...")
                shap.summary_plot(
                    shap_values,
                    c_features_shap,
                    title="Impact of Variables on Readmission Prediction",
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
                for pt_num in shap_indices: # this version uses the standard base value
                    print("Making force plot for patient ", pt_num, "...")
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[pt_num, :], c_features_shap.iloc[pt_num, : ],  # grabs the identities and values of components
                        text_rotation=30,  # easier to read
                        matplotlib=True,  # instead of Javascript
                        show=False,  # allows saving, etc.
                        link="logit"
                    )
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
                #         # link=shap.LogitLink()
                #     )
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


if __name__ == "__main__":
    classifiermany()
