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
from zz_generalHelpers import *
from zz_lgbmHelpers import lgbmClassificationHelpers
from zz_shapHelpers import shapHelpers

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
    debug = debug
    if trainmodels:
        for target in test_targets:

            figfolder = make_figfolder_for_target(debug, target)
            datafolder = make_datafolder_for_target(debug, target)
            modelfolder = make_modelfolder_for_target(debug, target)
            tablefolder = make_report_tables_folder(debug)

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

            train_labels = data_train[target]
            test_labels = data_test[target]
            valid_labels = data_valid[target]
            labels = data[target]

            # Set threshold based on prevalence of positive target
            negativeclassnum = len(data[data[target] == 0])
            positiveclassnum = len(data[data[target] == 1])
            proportionposneg = positiveclassnum / negativeclassnum
            class_thresh = proportionposneg
            print(f"Threshold for {target} is {class_thresh}")

            if target in configcols.SILLY_TARGETS:
                class_thresh = 0.5
                print(f"Just kidding, threshold for {target} is {class_thresh}")

            train_features = data_train.drop(
                configcols.C_READMIT_DROP_COLS, axis=1, errors="ignore"
            )
            train_features = train_features.drop(
                configcols.DATETIME_COLS, axis=1, errors="ignore"
            )

            test_features = data_test.drop(
                configcols.C_READMIT_DROP_COLS, axis=1, errors="ignore"
            )
            test_features = test_features.drop(
                configcols.DATETIME_COLS, axis=1, errors="ignore"
            )

            valid_features = data_valid.drop(
                configcols.C_READMIT_DROP_COLS, axis=1, errors="ignore"
            )
            valid_features = valid_features.drop(
                configcols.DATETIME_COLS, axis=1, errors="ignore"
            )

            features = data.drop(
                configcols.C_READMIT_DROP_COLS, axis=1, errors="ignore"
            )
            features = features.drop(configcols.DATETIME_COLS, axis=1, errors="ignore")

            if target == "financialclass_binary":
                print(f"Target is {target}, dropping raw insurance column...")
                train_features = train_features.drop(
                    "insurance2", axis=1, errors="ignore"
                )
                test_features = test_features.drop(
                    "insurance2", axis=1, errors="ignore"
                )
                valid_features = valid_features.drop(
                    "insurance2", axis=1, errors="ignore"
                )
                features = features.drop("insurance2", axis=1, errors="ignore")

            if target == "discharged_in_past_30d":
                train_features = train_features.drop(
                    [
                        "discharged_in_past_30d",
                        "days_between_current_admission_and_previous_discharge",
                    ],
                    axis=1,
                    errors="ignore",
                )
                test_features = test_features.drop(
                    [
                        "discharged_in_past_30d",
                        "days_between_current_admission_and_previous_discharge",
                    ],
                    axis=1,
                    errors="ignore",
                )
                valid_features = valid_features.drop(
                    [
                        "discharged_in_past_30d",
                        "days_between_current_admission_and_previous_discharge",
                    ],
                    axis=1,
                    errors="ignore",
                )
                features = features.drop(
                    [
                        "discharged_in_past_30d",
                        "days_between_current_admission_and_previous_discharge",
                    ],
                    axis=1,
                    errors="ignore",
                )
            elif target != "discharged_in_past_30d":
                pass

            if target in configcols.C_READMIT_BINARY_TARGETS:
                readmit_targets = configcols.C_READMIT_BINARY_TARGETS
                train_features = train_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                test_features = test_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                valid_features = valid_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                features = features.drop(readmit_targets, axis=1, errors="ignore")

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
                    train_features = train_features.drop(col, axis=1, errors="ignore")
                    test_features = test_features.drop(col, axis=1, errors="ignore")

                    valid_features = valid_features.drop(col, axis=1, errors="ignore")
                    features = features.drop(col, axis=1, errors="ignore")

            elif target == "gender_binary":
                gender_dependent = configcols.GENDER_DEPENDENT
                train_features = train_features.drop(
                    gender_dependent, axis=1, errors="ignore"
                )
                test_features = test_features.drop(
                    gender_dependent, axis=1, errors="ignore"
                )
                valid_features = valid_features.drop(
                    gender_dependent, axis=1, errors="ignore"
                )
                features = features.drop(gender_dependent, axis=1, errors="ignore")

            elif target == "race_binary":
                race_cols = configcols.RACE_COLS
                train_features = train_features.drop(race_cols, axis=1, errors="ignore")
                test_features = test_features.drop(race_cols, axis=1, errors="ignore")
                valid_features = valid_features.drop(race_cols, axis=1, errors="ignore")
                features = features.drop(race_cols, axis=1, errors="ignore")

            elif target == "financialclass_binary":
                financial_cols = configcols.FINANCIAL_COLS
                train_features = train_features.drop(
                    financial_cols, axis=1, errors="ignore"
                )
                test_features = test_features.drop(
                    financial_cols, axis=1, errors="ignore"
                )
                valid_features = valid_features.drop(
                    financial_cols, axis=1, errors="ignore"
                )
                features = features.drop(financial_cols, axis=1, errors="ignore")
                readmit_targets = configcols.C_READMIT_BINARY_TARGETS
                train_features = train_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                test_features = test_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                valid_features = valid_features.drop(
                    readmit_targets, axis=1, errors="ignore"
                )
                features = features.drop(readmit_targets, axis=1, errors="ignore")

            file_title = f"{target} test_features "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            ext = ".pickle"
            title = file_title + timestr + ext
            data_file = datafolder / title
            test_features.to_pickle(data_file)

            file_title = f"{target} test_labels "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            data_file = datafolder / title
            test_labels.to_pickle(data_file)

            file_title = f"{target} features "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            data_file = datafolder / title
            features.to_pickle(data_file)

            feature_list = list(train_features)
            df = pd.DataFrame(feature_list, columns=["features"])
            spreadsheet_title = f"Feature list 05 {target} "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".csv"
            title = spreadsheet_title + timestr + ext
            feature_list_file = datafolder / title
            df.to_csv(feature_list_file, index=False)

            print("Predicting", target)
            d_train = lgb.Dataset(
                train_features, label=train_labels, free_raw_data=True
            )
            d_test = lgb.Dataset(
                test_features, label=test_labels, reference=d_train, free_raw_data=True
            )
            d_valid = lgb.Dataset(
                valid_features,
                label=valid_labels,
                reference=d_train,
                free_raw_data=True,
            )
            # evals_result = {}  # to record eval results for plotting
            params = config.C_READMIT_PARAMS_LGBM
            early_stopping_rounds = 200
            if debug:
                early_stopping_rounds = 10

            # LGB wrapper doesn't play as nicely with calibration, bc no "predict_proba" directly

            # gbm_model = lgb.train(
            #     params,
            #     d_train,
            #     num_boost_round=10_000_000,
            #     valid_sets=[d_test, d_train],
            #     valid_names=["test", "train"],
            #     feval=lgb_f1_score,
            #     evals_result=evals_result,
            #     early_stopping_rounds=early_stopping_rounds,
            #     verbose_eval=25,
            # )

            # gbm_model = lgb.LGBMClassifier(
            #     boosting_type="gbdt",
            #     colsample_bytree=0.707_630_032_256_903,
            #     is_unbalance="false",
            #     learning_rate=0.010_302_298_912_236_304,
            #     max_depth=-1,
            #     min_child_samples=360,
            #     min_child_weight=0.001,
            #     min_split_gain=0.0,
            #     n_estimators=568,
            #     n_jobs=-1,
            #     num_leaves=99,
            #     num_rounds=10_000_000,
            #     objective="binary",
            #     predict_contrib=True,
            #     random_state=config.SEED,
            #     reg_alpha=0.592_673_416_782_159_5,
            #     reg_lambda=0.149_874_982_676_853_4,
            #     silent=False,
            #     subsample_for_bin=240_000,
            #     subsample_freq=0,
            #     subsample=0.602_760_991_384_907_5,
            # )

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
                eval_metric="l1",
                early_stopping_rounds=early_stopping_rounds,
            )

            figfolder = make_figfolder_for_target(debug, target)
            datafolder = make_datafolder_for_target(debug, target)
            modelfolder = make_modelfolder_for_target(debug, target)
            tablefolder = make_report_tables_folder(debug)
            evals_result = gbm_model._evals_result
            # Initialize lgbmHelpers
            calibrate_please = True
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
                calibrate_please,
            )
            metricsgen.lgbm_save_ttv_split() # make sure to save the ttv first
            # it uses a "write" command for the h5 file, and will overwrite any other file with the same name
            
            # Save LGBM model to file and return the pickled model's file name
            pkl_model = metricsgen.lgbm_save_model_to_pkl_and_h5()

            if generatemetrics:
                metricsgen.lgbm_save_feature_importance_plot()
                metricsgen.lgbm_classification_results()

            if generateshap:
                # Load model and generate SHAP values

                with open(pkl_model, "rb") as fin:
                    gbm_model = pickle.load(fin)
                print(f"{target} Generating SHAP values...")
                explainer = shap.TreeExplainer(gbm_model)
                features_shap = features.sample(
                    n=20000, random_state=seed, replace=False
                )

                shap_values = explainer.shap_values(features_shap)

                # The shapHelpers methods use predefined folders to keep everything organized
                # Make sure they exist
                figfolder = make_figfolder_for_target(debug, target)
                datafolder = make_datafolder_for_target(debug, target)
                modelfolder = make_modelfolder_for_target(debug, target)

                # Then initialize shapHelpers
                helpshap = shapHelpers(
                    target,
                    features_shap,
                    shap_values,
                    figfolder,
                    datafolder,
                    modelfolder,
                )
                # and prosper!

                # Save raw shap values to disk as pickled np array
                helpshap.shap_save_to_disk()

                # Make CSV and pickle of SHAP values in descending order of importance
                helpshap.shap_save_ordered_values()

                # Change the column names so the figures look nice
                # LightGBM doesn't like spaces in column names, so it's best to train
                # with the underscored column names and then change them for plotting
                helpshap.shap_prettify_column_names(
                    prettycols_file=config.PRETTIFYING_COLUMNS_CSV
                )

                # Make bar and dot summary plots
                helpshap.shap_plot_summaries(
                    title_in_figure="Impact of Variables on Readmission Prediction"
                )

                # Make a specified number of force plots for randomly selected rows
                helpshap.shap_random_force_plots(
                    n_plots=5, expected_value=explainer.expected_value
                )

                # Make dependence plots for a specified number of top variables
                helpshap.shap_top_dependence_plots(n_plots=10)


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
#                 features_shap,
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
