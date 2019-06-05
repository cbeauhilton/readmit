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
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
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
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return "f1", f1_score(y_true, y_hat), True


def regressormany(
    test_targets,
    debugfraction=0.005,
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
                data = pd.read_pickle(filename)
                data_train = pd.read_pickle(train_file)
                data_test = pd.read_pickle(test_set_file)
                data_valid = pd.read_pickle(valid_set_file)
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

            if target == "length_of_stay_in_days":
                print(target, "== LoS cleaning")
                data_train = data_train[data_train["length_of_stay_in_days"] < 31]
                data_test = data_test[data_test["length_of_stay_in_days"] < 31]
                data_valid = data_valid[data_valid["length_of_stay_in_days"] < 31]
                data = data[data["length_of_stay_in_days"] < 31]
                data_train = data_train[data_train["length_of_stay_in_days"] > 1]
                data_test = data_test[data_test["length_of_stay_in_days"] > 1]
                data_valid = data_valid[data_valid["length_of_stay_in_days"] > 1]
                data = data[data["length_of_stay_in_days"] > 1]
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

            elif target == "days_between_current_discharge_and_next_admission":
                print(
                    target,
                    "== readmit cleaning (drop nulls and >30 days until readmission)",
                )
                data_train = data_train[
                    data_train["days_between_current_discharge_and_next_admission"] > 0
                ]
                data_test = data_test[
                    data_test["days_between_current_discharge_and_next_admission"] > 0
                ]
                data_valid = data_valid[
                    data_valid["days_between_current_discharge_and_next_admission"] > 0
                ]
                data = data[
                    data["days_between_current_discharge_and_next_admission"] > 0
                ]
                data_train = data_train[
                    data_train["days_between_current_discharge_and_next_admission"] < 30
                ]
                data_test = data_test[
                    data_test["days_between_current_discharge_and_next_admission"] < 30
                ]
                data_valid = data_valid[
                    data_valid["days_between_current_discharge_and_next_admission"] < 30
                ]
                data = data[
                    data["days_between_current_discharge_and_next_admission"] < 30
                ]
                data_train = data_train[
                    data_train[
                        "days_between_current_discharge_and_next_admission"
                    ].notnull()
                ]
                data_test = data_test[
                    data_test[
                        "days_between_current_discharge_and_next_admission"
                    ].notnull()
                ]
                data_valid = data_valid[
                    data_valid[
                        "days_between_current_discharge_and_next_admission"
                    ].notnull()
                ]
                data = data[
                    data["days_between_current_discharge_and_next_admission"].notnull()
                ]
                print("Length of data with readmit >0, <100, not null:", len(data))

            elif target == "patient_age":
                print(target, "== age cleaning, if it ever becomes necessary")
                data = data[data.patient_age < 100]
                data_train = data_train[data_train.patient_age < 100]
                data_test = data_test[data_test.patient_age < 100]
                data_valid = data_valid[data_valid.patient_age < 100]
                data = data[data.patient_age >= 18]
                data_train = data_train[data_train.patient_age >= 18]
                data_test = data_test[data_test.patient_age >= 18]
                data_valid = data_valid[data_valid.patient_age >= 18]

            c_train_labels = data_train[target]
            c_test_labels = data_test[target]
            c_valid_labels = data_valid[target]

            c_train_features = data_train
            c_test_features = data_test
            c_valid_features = data_valid
            c_features = data

            # c_train_features = data_train.drop(target, axis=1)
            # c_test_features = data_test.drop(target, axis=1)
            # c_valid_features = data_valid.drop(target, axis=1)
            # c_features = data.drop(target, axis=1)

            date_cols = configcols.DATETIME_COLS
            discharge_lab_cols = configcols.DISCHARGE_LAB_COLS
            discharge_other_cols = configcols.DISCHARGE_OTHER_COLS
            gen_drop_cols = configcols.C_READMIT_DROP_COLS
            los_dependent = configcols.LOS_DEPENDENT
            los_targets = configcols.C_LOS_BINARY_TARGETS
            readmit_cols = configcols.C_READMIT_BINARY_TARGETS

            if target == "length_of_stay_in_days":
                print(target, "== LoS, dropping cols")

                all_the_cols = [
                    date_cols,
                    discharge_lab_cols,
                    discharge_other_cols,
                    gen_drop_cols,
                    los_dependent,
                    los_targets,
                    readmit_cols,
                    target,
                ]

                for col in all_the_cols:

                    c_train_features = c_train_features.drop(
                        col, axis=1, errors="ignore"
                    )
                    c_test_features = c_test_features.drop(col, axis=1, errors="ignore")

                    c_valid_features = c_valid_features.drop(
                        col, axis=1, errors="ignore"
                    )
                    c_features = c_features.drop(col, axis=1, errors="ignore")

            elif target == "days_between_current_discharge_and_next_admission":
                print(target, " == readmission, dropping cols")

                all_the_cols = [date_cols, gen_drop_cols, readmit_cols, target]

                for col in all_the_cols:

                    c_train_features = c_train_features.drop(
                        col, axis=1, errors="ignore"
                    )
                    c_test_features = c_test_features.drop(col, axis=1, errors="ignore")

                    c_valid_features = c_valid_features.drop(
                        col, axis=1, errors="ignore"
                    )
                    c_features = c_features.drop(col, axis=1, errors="ignore")

            elif target == "patient_age":
                print(target, " == patient_age, dropping cols")

                all_the_cols = [date_cols, gen_drop_cols, target]

                for col in all_the_cols:

                    c_train_features = c_train_features.drop(
                        col, axis=1, errors="ignore"
                    )
                    c_test_features = c_test_features.drop(col, axis=1, errors="ignore")

                    c_valid_features = c_valid_features.drop(
                        col, axis=1, errors="ignore"
                    )
                    c_features = c_features.drop(col, axis=1, errors="ignore")

            file_title = f"{target} c_test_features "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / "debug" / target
                if not os.path.exists(datafolder):
                    print("Making folder called", datafolder)
                    os.makedirs(datafolder)
            else:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / target
                if not os.path.exists(datafolder):
                    print("Making folder called", datafolder)
                    os.makedirs(datafolder)
            data_file = datafolder / title
            c_test_features.to_pickle(data_file)

            file_title = f"{target} c_test_labels "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / "debug" / target
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
                datafolder = config.PROCESSED_DATA_DIR / timestrfolder / "debug" / target
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
            c_params = config.R_READMIT_PARAMS_LGBM
            early_stopping_rounds = 200
            if debug:
                early_stopping_rounds = 10
            c_gbm = lgb.train(
                c_params,
                c_d_train,
                num_boost_round=10_000_000,
                valid_sets=[
                    c_d_test,
                    # c_d_train
                ],
                valid_names=[
                    "test",
                    # "train"
                ],
                early_stopping_rounds=early_stopping_rounds,
                evals_result=c_evals_result,
                verbose_eval=25,
            )

            print("Dumping model with pickle...")
            file_title = f"{target} MODEL "
            timestr = time.strftime("%Y-%m-%d-%H%M")
            timestrfolder = time.strftime("%Y-%m-%d")
            ext = ".pickle"
            title = file_title + timestr + ext
            if debug:
                modelfolder = config.MODELS_DIR / timestrfolder / "debug" / target
                if not os.path.exists(modelfolder):
                    print("Making folder called", modelfolder)
                    os.makedirs(modelfolder)
            else:
                modelfolder = config.MODELS_DIR / timestrfolder / target
                if not os.path.exists(modelfolder):
                    print("Making folder called", modelfolder)
                    os.makedirs(modelfolder)
            pkl_model = modelfolder / title
            with open(pkl_model, "wb") as fout:
                pickle.dump(c_gbm, fout)
            if generatemetrics:
                print("Generating results, tables, and figures...")

                print("Plotting metrics recorded during training...")
                ax = lgb.plot_metric(c_evals_result, metric="rmse")
                # plt.show()
                plt.close()

                print("Plot feature importances...")
                ax = lgb.plot_importance(c_gbm, figsize=(5, 20))
                figure_title = f"{target} Feature Importances "
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
                plt.close()

                # Generate Metrics
                print("Generating metrics...")
                c_predict_labels = c_gbm.predict(c_test_features, num_iteration=c_gbm.best_iteration)
                RMSE = mean_squared_error(c_test_labels, c_predict_labels) ** 0.5
                MEANAE = mean_absolute_error(c_test_labels, c_predict_labels)
                MEDIANAE = median_absolute_error(c_test_labels, c_predict_labels)
                RSQUARED = r2_score(c_test_labels, c_predict_labels)
                EV = explained_variance_score(c_test_labels, c_predict_labels)

                # R Squared plot
                print("Plotting R squared...")
                plt.scatter(c_test_labels, c_predict_labels)
                plt.xlabel('Actual values')
                plt.ylabel('Predicted values')
                plt.plot(np.unique(c_test_labels), np.poly1d(np.polyfit(c_test_labels, c_predict_labels, 1))(np.unique(c_test_labels)))
                plt.text(0.6, 0.5, 'R-squared = %0.2f' % RSQUARED)
                figure_title = f"{target} R Squared "
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
                plt.close()

                # make CSV of metrics
                d = [[timestr, target, RMSE, MEANAE, MEDIANAE, RSQUARED, EV]]
                df = pd.DataFrame(
                    d,
                    columns=(
                        "Time",
                        "Target",
                        "RMSE",
                        "Median Absolute Error",
                        "Mean Absolute Error",
                        "R Squared",
                        "Explained Variance"
                    ),
                )
                if debug:
                    df.to_csv("regressordebugreports.csv", mode="a", header=True)
                else:
                    df.to_csv(config.REGRESSOR_TRAINING_REPORTS, mode="a", header=True)

            if generateshap:
                # load GBM model
                with open(pkl_model, "rb") as fin:
                    c_gbm = pickle.load(fin)

                print(f"{target} Generating SHAP values...")
                startTime1 = datetime.now()
                explainer = shap.TreeExplainer(c_gbm)
                c_features_shap = c_features.sample(
                    n=20000, random_state=seed, replace=False
                )
                shap_values = explainer.shap_values(c_features_shap)
                endTime1 = datetime.now()
                print("SHAP values generated in: ", endTime1 - startTime1)

                print("Saving to disk...")  # will be a pickled numpy array
                file_title = f"{target} SHAP "
                timestr = time.strftime("%Y-%m-%d-%H%M")
                timestrfolder = time.strftime("%Y-%m-%d")
                ext = ".pickle"
                title = file_title + timestr + ext
                if debug:
                    modelfolder = config.MODELS_DIR / timestrfolder / "debug" / target
                    if not os.path.exists(modelfolder):
                        print("Making folder called", modelfolder)
                        os.makedirs(modelfolder)
                else:
                    modelfolder = config.MODELS_DIR / timestrfolder / target
                    if not os.path.exists(modelfolder):
                        print("Making folder called", modelfolder)
                        os.makedirs(modelfolder)
                full_path = modelfolder / title
                with open(full_path, "wb") as f:
                    pickle.dump(shap_values, f)

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
                
                pickle_title = f"{target}_{timestr}_shap_df.pickle"
                df_shap_train.to_pickle(datafolder / pickle_title)
                imp_cols = pd.DataFrame(imp_cols)
                print(imp_cols)
                csv_title = f"{target}_{timestr}_shap.csv"
                imp_cols.to_csv(datafolder / csv_title)
                
                # Make the feature names look nice for publication
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

                # make SHAP plots
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
                for pt_num in shap_indices:  # this version uses the standard base value
                    print("Making force plot for patient ", pt_num, "...")
                    shap.force_plot(
                        explainer.expected_value,
                        shap_values[pt_num, :],
                        c_features_shap.iloc[
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




                # Dependence plots don't like the "category" dtype, will only work with
                # category codes (ints).
                # Make a copy of the df that uses cat codes
                # Then use the original df as the "display"
                df_with_codes = c_features_shap.copy()
                for col in list(df_with_codes.select_dtypes(include="category")):
                    df_with_codes[col] = df_with_codes[col].cat.codes

                for i in range(10):
                    shap.dependence_plot(
                        "rank(%d)" % i,
                        shap_values,
                        df_with_codes,
                        display_features=c_features_shap,
                        show=False,
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

                # How long did this take?
                print("This program,", os.path.basename(__file__), "took")
                print(datetime.now() - startTime)
                print("to run.")


if __name__ == "__main__":
    regressormany()
