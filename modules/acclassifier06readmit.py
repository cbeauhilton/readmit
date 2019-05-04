import os
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import shap
from sklearn.exceptions import UndefinedMetricWarning

import config
import configcols
from zz_generalHelpers import (
    lgb_f1_score,
    make_datafolder_for_target,
    make_figfolder_for_target,
    make_modelfolder_for_target,
    make_report_tables_folder,
    train_test_valid_80_10_10_split,
)
from zz_lgbmHelpers import lgbmClassificationHelpers
from zz_shapHelpers import shapHelpers

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

try:
    import cPickle as pickle
except BaseException:
    import pickle

print("About to run", os.path.basename(__file__))
startTime = datetime.now()


increasingly_parsimonious = ["full", "ccf", "community"]

for dataset in increasingly_parsimonious:

    startTime1 = datetime.now()

    if dataset is "full":
        print(dataset)
        rename_dict = {
            "readmitted30d": "readmitted30d_full",
            "length_of_stay_over_7_days": "length_of_stay_over_7_days_full",
            "financialclass_binary": "financialclass_binary_full",
            "gender_binary": "gender_binary_full",
            "race_binary": "race_binary_full",
        }
        targets = [
            "readmitted30d_full",
            "length_of_stay_over_7_days_full",
            "financialclass_binary_full",
            "gender_binary_full",
            "race_binary_full",
        ]

    elif dataset is "ccf":
        print(dataset)
        rename_dict = {
            "readmitted30d": "readmitted30d_ccf",
            "length_of_stay_over_7_days": "length_of_stay_over_7_days_ccf",
            "financialclass_binary": "financialclass_binary_ccf",
            "gender_binary": "gender_binary_ccf",
            "race_binary": "race_binary_ccf",
        }
        targets = [
            "readmitted30d_ccf",
            "length_of_stay_over_7_days_ccf",
            "financialclass_binary_ccf",
            "gender_binary_ccf",
            "race_binary_ccf",
        ]

    elif dataset is "community":
        print(dataset)
        rename_dict = {
            "readmitted30d": "readmitted30d_community",
            "length_of_stay_over_7_days": "length_of_stay_over_7_days_community",
            "financialclass_binary": "financialclass_binary_community",
            "gender_binary": "gender_binary_community",
            "race_binary": "race_binary_community",
        }
        targets = [
            "readmitted30d_community",
            "length_of_stay_over_7_days_community",
            "financialclass_binary_community",
            "gender_binary_community",
            "race_binary_community",
        ]

    for target in targets:

        filename = config.PROCESSED_FINAL
        print("Loading", filename)
        data = pd.read_pickle(filename)
        print("File loaded.")
        print("Dataset:", dataset)

        if target == "readmitted30d_full":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.READMITTED30D_SHAP_COLS_FULL]
            data = data[data["dischargedispositiondescription"] != "Expired"]
            name_for_figs = "Readmission"

        elif target == "readmitted30d_ccf":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.READMITTED30D_SHAP_COLS_CCF]
            data = data[data["dischargedispositiondescription"] != "Expired"]
            name_for_figs = "Readmission"

        elif target == "readmitted30d_community":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.READMITTED30D_SHAP_COLS_COMMUNITY]
            data = data[data["dischargedispositiondescription"] != "Expired"]
            name_for_figs = "Readmission"

        elif target == "length_of_stay_over_7_days_full":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.LOS7_SHAP_COLS_FULL]
            data = data[data["patientclassdescription"] != "Observation"]
            name_for_figs = "Length of Stay"

        elif target == "length_of_stay_over_7_days_ccf":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.LOS7_SHAP_COLS_CCF]
            data = data[data["patientclassdescription"] != "Observation"]
            name_for_figs = "Length of Stay"

        elif target == "length_of_stay_over_7_days_community":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.LOS7_SHAP_COLS_COMMUNITY]
            data = data[data["patientclassdescription"] != "Observation"]
            name_for_figs = "Length of Stay"

        elif target == "gender_binary_full":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.GENDER_SHAP_COLS_FULL]
            name_for_figs = "Gender"

        elif target == "gender_binary_ccf":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.GENDER_SHAP_COLS_CCF]
            name_for_figs = "Gender"

        elif target == "gender_binary_community":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.GENDER_SHAP_COLS_COMMUNITY]
            name_for_figs = "Gender"

        elif target == "race_binary_full":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.RACE_SHAP_COLS_FULL]
            name_for_figs = "Race"

        elif target == "race_binary_ccf":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.RACE_SHAP_COLS_CCF]
            name_for_figs = "Race"

        elif target == "race_binary_community":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.RACE_SHAP_COLS_COMMUNITY]
            name_for_figs = "Race"

        elif target == "financialclass_binary_full":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.INSURANCE_SHAP_COLS_FULL]
            name_for_figs = "Insurance"

        elif target == "financialclass_binary_ccf":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.INSURANCE_SHAP_COLS_CCF]
            name_for_figs = "Insurance"

        elif target == "financialclass_binary_community":
            data = data.rename(columns=rename_dict)
            print(target)
            data = data[configcols.INSURANCE_SHAP_COLS_COMMUNITY]
            name_for_figs = "Insurance"

        seed = config.SEED
        debug = False
        print("Debug:", debug)

        figfolder = make_figfolder_for_target(debug, target)
        datafolder = make_datafolder_for_target(debug, target)
        modelfolder = make_modelfolder_for_target(debug, target)
        tablefolder = make_report_tables_folder(debug)

        # For financial class, gender, race, some are missing
        # data = data[data[target].notnull()] # added this line to the ttv split script

        train_set, test_set, valid_set = train_test_valid_80_10_10_split(
            data, target, seed
        )

        train_labels = train_set[target]
        train_features = train_set.drop([target], axis=1)

        test_labels = test_set[target]
        test_features = test_set.drop([target], axis=1)

        valid_labels = valid_set[target]
        valid_features = valid_set.drop([target], axis=1)

        labels = data[target]
        features = data.drop([target], axis=1)

        print("Predicting", target)
        d_train = lgb.Dataset(train_features, label=train_labels, free_raw_data=True)

        d_test = lgb.Dataset(
            test_features, label=test_labels, reference=d_train, free_raw_data=True
        )

        d_valid = lgb.Dataset(
            valid_features, label=valid_labels, reference=d_train, free_raw_data=True
        )

        class_thresh = 0.5

        if (
            target is "readmitted30d_full"
            or "readmitted30d_ccf"
            or "readmitted30d_community"
            or "length_of_stay_over_7_days_ccf"
            or "length_of_stay_over_7_days_ccf"
            or "length_of_stay_over_7_days_community"
        ):
            class_thresh = 0.2
            print("class_thresh :", class_thresh)

        params = config.C_READMIT_PARAMS_LGBM
        early_stopping_rounds = 200
        # gbm_model = lgb.LGBMClassifier(**params) # <-- could also do this, but it's kind of nice to have it all explicit
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
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds,
        )
        evals_result = gbm_model._evals_result

        metricsgen = lgbmClassificationHelpers(
            target,
            class_thresh,
            gbm_model,
            evals_result,
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
            calibrate_please=True,
        )

        pkl_model = metricsgen.lgbm_save_model_to_pkl_and_h5()
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
        helpshap.shap_save_ordered_values()
        helpshap.shap_prettify_column_names(
            prettycols_file=config.PRETTIFYING_COLUMNS_CSV
        )
        helpshap.shap_plot_summaries(
            title_in_figure=f"Impact of Variables on {name_for_figs} Prediction"
        )
        helpshap.shap_random_force_plots(
            n_plots=5, expected_value=explainer.expected_value
        )
        helpshap.shap_top_dependence_plots(n_plots=10)

        print("This loop ran in:")
        print(datetime.now() - startTime1)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")


# data = data.rename(
#     index=str,
#     columns={
#         "readmitted30d": "readmitted30d_full",
#         "length_of_stay_over_7_days": "length_of_stay_over_7_days_full",
#         "financialclass_binary": "financialclass_binary_full",
#         "gender_binary": "gender_binary_full",
#         "race_binary": "race_binary_full",
#     },
# )
# data = data.rename(
#     index=str,
#     columns={
#         "readmitted30d": "readmitted30d_ccf",
#         "length_of_stay_over_7_days": "length_of_stay_over_7_days_ccf",
#         "financialclass_binary": "financialclass_binary_ccf",
#         "gender_binary": "gender_binary_ccf",
#         "race_binary": "race_binary_ccf",
#     },
# )

# data = data.rename(
#     index=str,
#     columns={
#         "readmitted30d": "readmitted30d_community",
#         "length_of_stay_over_7_days": "length_of_stay_over_7_days_community",
#         "financialclass_binary": "financialclass_binary_community",
#         "gender_binary": "gender_binary_community",
#         "race_binary": "race_binary_community",
#     },
# )

