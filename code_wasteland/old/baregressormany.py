import sys

sys.path.append("modules")

from datetime import datetime

startTime = datetime.now()

import os
import pandas as pd
import lightgbm as lgb
import time
import numpy as np
try:
    import cPickle as pickle
except BaseException:
    import pickle

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import z_compare_auc_delong_xu
import scipy.stats
from scipy import stats

import config

pd.options.display.max_columns = 2000

# Load file
def regressormany(
    test_targets,
    debugfraction=0.005,
    debug=False,
    trainmodels=False,
    generateshap=False,
    generatemetrics=False,
):
    test_targets = test_targets
    debugfraction = debugfraction
    filename = config.PROCESSED_FINAL
    print("Loading", filename)
    data = pd.read_pickle(filename)

    print("File loaded.")

    if trainmodels:
        print("Building train and test...")
        data_train = data[
            (data["admissiontime"] > config.TRAIN_START)
            & (data["admissiontime"] < config.TRAIN_END)
        ]

        data_test = data[
            (data["admissiontime"] > config.TEST_START)
            & (data["admissiontime"] < config.TEST_END)
        ]

        data_valid = data[
            (data["admissiontime"] > config.VALID_START)
            & (data["admissiontime"] < config.VALID_END)
        ]
        if debug:
            print(f"Selecting small portion ({debugfraction}) for debugging...", )
            data_train = data_train.sample(frac=debugfraction, random_state=config.SEED)
            data_test = data_test.sample(frac=debugfraction, random_state=config.SEED)
            data_valid = data_valid.sample(frac=debugfraction, random_state=config.SEED)
            data = data.sample(frac=0.05, random_state=config.SEED)
        else:
            pass
        data_train = data_train.drop(configcols.R_LOS_DROP_COLS_LGBM, axis=1)
        data_test = data_test.drop(configcols.R_LOS_DROP_COLS_LGBM, axis=1)
        data_valid = data_valid.drop(configcols.R_LOS_DROP_COLS_LGBM, axis=1)
        data = data.drop(configcols.R_LOS_DROP_COLS_LGBM, axis=1)
        for target in test_targets:
            r_train_labels = data_train[target]
            r_train_features = data_train.drop(test_targets, axis=1)
            r_test_labels = data_test[target]
            r_test_features = data_test.drop(test_targets, axis=1)
            r_valid_labels = data_valid[target]
            r_valid_features = data_valid.drop(test_targets, axis=1)
            r_features = data.drop(test_targets, axis=1)

            # print(list(r_train_features))
            print("Predicting", target)
            r_d_train = lgb.Dataset(r_train_features, label=r_train_labels, free_raw_data=True)
            r_d_test = lgb.Dataset(
                r_test_features, label=r_test_labels, reference=r_d_train, free_raw_data=True
            )
            r_d_valid = lgb.Dataset(
                r_valid_features, label=r_valid_labels, reference=r_d_train, free_raw_data=True
            )
            # r_evals_result = {}  # to record eval results for plotting
            r_params = config.R_READMIT_PARAMS_LGBM
            early_stopping_rounds = 100
            if debug:
                early_stopping_rounds = 10
            startTimetrain = datetime.now()
            print("Training starting at ", startTimetrain)
            r_gbm = lgb.train(r_params, 
                            r_d_train, 
                            100000, 
                            valid_sets=r_d_test, 
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=25,
                            # evals_result=r_evals_result,
                        )
            endTimetrain = datetime.now()
            traintiming = endTimetrain - startTimetrain
            print("Training completed in: ", traintiming)
            print("Dumping model with pickle...")
            file_title = f"{target} MODEL "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            modeldir = config.MODELS_DIR
            if debug:
                modelfolder =  modeldir / timestrfolder / "debug"
                if not os.path.exists(modelfolder):
                    print("Making folder called", modelfolder)
                    os.makedirs(modelfolder)
            else:
                modelfolder = modeldir / timestrfolder
                if not os.path.exists(modelfolder):
                    print("Making folder called", modelfolder)
                    os.makedirs(modelfolder)
            pkl_model = modelfolder / title
            with open(pkl_model, "wb") as fout:
                pickle.dump(r_gbm, fout)
            if generateshap:
                with open(pkl_model, "rb") as fin:
                    r_gbm = pickle.load(fin)

                print("Generating SHAP values...")
                startTimeshap = datetime.now()
                print(startTimeshap)
                explainer = shap.TreeExplainer(r_gbm)
                shap_values = explainer.shap_values(r_features)
                endTimeshap = datetime.now()
                shaptiming = endTimeshap - startTimeshap
                print("SHAP values generated in: ", shaptiming)
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
                    r_features,
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
                    r_features,
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
                for pt_num in shap_indices:
                    print("Making force plot for patient", pt_num, "...")
                    shap.force_plot(
                        0,  # set expected value to zero
                        np.hstack(
                            [shap_values[pt_num, :], explainer.expected_value]
                        ),  # add expected value as a bar in the force plot
                        pd.concat(
                            [
                                r_features,
                                pd.DataFrame(
                                    {"Base value": [explainer.expected_value]}
                                ),
                            ],
                            axis=1,
                        ).iloc[
                            pt_num, :
                        ],  # grabs the identities and values of components
                        text_rotation=30,  # easier to read
                        matplotlib=True,  # instead of Javascript
                        show=False,  # allows saving, etc.
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
            if generatemetrics:
                print("Generating results, tables, and figures...")
                print("Plot feature importances...")
                ax = lgb.plot_importance(
                    r_gbm, figsize=(5, 20), importance_type="gain", precision=2
                )
                figure_title = f"{target} Feature Importances"
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

                # print("Generating ROC curve...")
                # # plt.figure(figsize=(5,5))
                # plt.title("Receiver Operating Characteristic")
                # plt.plot(fpr, tpr, "b", label="AUC ="
                # plt.legend(loc="lower right")
                # plt.plot([0, 1], [0, 1], "r--")
                # plt.xlim([-0.011, 1.011])
                # plt.ylim([-0.011, 1.011])
                # plt.ylabel("True Positive Rate")
                # plt.xlabel("False Positive Rate")
                # figure_title = (
                #     f"{target} Receiver_Operating_Characteristic "
                # )
                # timestr = time.strftime("%Y-%m-%d-%H%M")
                # ext = ".png"
                # title = figure_title + timestr + ext
                # plt.savefig(
                #     (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                # )
                # # plt.show()
                # plt.close()



                # conf_mx = metrics.confusion_matrix(r_test_labels, r_predict_labels)
                # fig = sns.heatmap(conf_mx, square=True, annot=True, fmt="d", cbar=False)
                # fig = fig.get_figure()
                # plt.xlabel("True Label")
                # plt.ylabel("Predicted Label")
                # figure_title = f"{target} Confusion Matrix AUC %0.2f_" %
                # timestr = time.strftime("%Y-%m-%d-%H%M")
                # ext = ".png"
                # title = figure_title + timestr + ext
                # plt.savefig(
                #     (figfolder / title), dpi=400, transparent=False, bbox_inches="tight"
                # )
                # plt.close()

                # conf_mx = pd.DataFrame(conf_mx)

                # FP = conf_mx.sum(axis=0) - np.diag(conf_mx)
                # FN = conf_mx.sum(axis=1) - np.diag(conf_mx)
                # TP = np.diag(conf_mx)
                # TN = conf_mx.values.sum() - (FP + FN + TP)

                # # Sensitivity, hit rate, recall, or true positive rate
                # TPR = TP / (TP + FN)
                # # Specificity or true negative rate
                # TNR = TN / (TN + FP)
                # # Precision or positive predictive value
                # PPV = TP / (TP + FP)
                # # Negative predictive value
                # NPV = TN / (TN + FN)
                # # Fall out or false positive rate
                # FPR = FP / (FP + TN)
                # # False negative rate
                # FNR = FN / (TP + FN)
                # # False discovery rate
                # FDR = FP / (TP + FP)

                # # Overall accuracy
                # ACC = (TP + TN) / (TP + FP + FN + TN)

                # print("TPR:", TPR)
                # print("TNR:", TNR)
                # print("PPV:", PPV)
                # print("NPV:", NPV)
                # print("FPR:", FPR)
                # print("FNR:", FNR)
                # print("FDR:", FDR)
                # print("ACC:", ACC)
                print("...complete.")
                # How long did this take?
                print("This program took")
                print(datetime.now() - startTime)
                print("to run.")





if __name__ == "__main__":
    regressormany()




