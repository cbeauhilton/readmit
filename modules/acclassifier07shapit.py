import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import UndefinedMetricWarning

import cbh.config as config
import configcols
from cbh.generalHelpers import (
    lgb_f1_score,
    make_datafolder_for_target,
    make_figfolder_for_target,
    make_modelfolder_for_target,
    make_report_tables_folder,
    train_test_valid_80_10_10_split,
)
from cbh.lgbmHelpers import lgbmClassificationHelpers
from cbh.shapHelpers import shapHelpers

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

try:
    import cPickle as pickle
except BaseException:
    import pickle

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

targets = [
    "readmitted30d",
    "readmitted5d",
    "readmitted7d",
    "readmitted3d",
    "length_of_stay_over_3_days",
    "length_of_stay_over_5_days",
    "length_of_stay_over_7_days",
    "financialclass_binary",
    "age_gt_10_y",
    "age_gt_30_y",
    "age_gt_65_y",
    "gender_binary",
    "race_binary",
    "discharged_in_past_30d",
    # "died_within_48_72h_of_admission_combined",
]

shap_indices = [500]  # 10, 20, 30, 40, 50, 60

# for last one, could use large number or "None"

for shap_index in shap_indices:

    startTime1 = datetime.now()

    for target in targets:
        # Load CSV of top SHAP values, and select the first n
        csv_dir = config.SHAP_CSV_DIR
        shap_file = f"{target}_shap.csv"
        print("SHAP CSV:", csv_dir / shap_file)
        top_shaps = pd.read_csv(csv_dir / shap_file)
        top_shaps = top_shaps.rename(index=str, columns={"0": "feature_names"})
        top_shaps = top_shaps[:shap_index]
        shap_list = top_shaps["feature_names"].tolist()
        shap_list.append(target)  # to make the labels and features sets
        if target == "length_of_stay_over_3_days" or "length_of_stay_over_5_days" or "length_of_stay_over_7_days":
            shap_list.append("length_of_stay_in_days")
        # print(shap_list)

        filename = config.PROCESSED_FINAL
        print("Loading", filename)
        data = pd.read_pickle(filename)

        print("File loaded.")

        seed = config.SEED
        debug = False
        print("Debug:", debug)

        figfolder = make_figfolder_for_target(debug, target)
        datafolder = make_datafolder_for_target(debug, target)
        modelfolder = make_modelfolder_for_target(debug, target)
        tablefolder = make_report_tables_folder(debug)
        if target == "readmitted30d":
            name_for_figs = "Readmission"
            print("Dropping expired patients...")
            data = data[data["dischargedispositiondescription"] != "Expired"]
            # data = data[data["days_between_current_admission_and_previous_discharge"] > 1]
        elif (
            target == "length_of_stay_over_7_days"
            or "length_of_stay_over_5_days"
            or "length_of_stay_over_3_days"
        ):
            name_for_figs = "Length of Stay"
            print("Dropping expired and obs patients...")
            data = data[data["dischargedispositiondescription"] != "Expired"]
            data = data[data["patientclassdescription"] != "Observation"]
        elif (
            target == "died_within_48_72h_of_admission_combined"
        ):
            name_for_figs = "Death within 48--72 Hours of Admission"
            # print("Dropping expired and obs patients for death prediction...")
            # data = data[data["dischargedispositiondescription"] != "Expired"]
            # data = data[data["patientclassdescription"] != "Observation"]
        elif target == "financialclass_binary":
            name_for_figs = "Insurance"
        elif target == "gender_binary":
            name_for_figs = "Gender"
        elif target == "race_binary":
            name_for_figs = "Race"
        elif target == "discharged_in_past_30d":
            name_for_figs == "Discharged in Past 30 Days"

        data = data[shap_list]

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

        if target == "length_of_stay_over_3_days" or "length_of_stay_over_5_days" or "length_of_stay_over_7_days":
            features = features.drop(["length_of_stay_in_days"], axis=1)
            train_features = train_features.drop(["length_of_stay_in_days"], axis=1)
            test_features = test_features.drop(["length_of_stay_in_days"], axis=1)
            valid_features = valid_features.drop(["length_of_stay_in_days"], axis=1)

        print("Predicting", target, "with", shap_index, "top features...")
        d_train = lgb.Dataset(train_features, label=train_labels, free_raw_data=True)

        d_test = lgb.Dataset(
            test_features, label=test_labels, reference=d_train, free_raw_data=True
        )

        d_valid = lgb.Dataset(
            valid_features, label=valid_labels, reference=d_train, free_raw_data=True
        )

        class_thresh = 0.5

        if target is "readmitted30d" or "length_of_stay_over_7_days":
            class_thresh = 0.2
            print("class_thresh changed to:", class_thresh)

        if target is "discharged_in_past_30d":
            class_thresh = 0.2
            print("class_thresh changed to:", class_thresh)

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
            calibrate_please=False,
        )
        metricsgen.lgbm_save_ttv_split()
        pkl_model = metricsgen.lgbm_save_model_to_pkl_and_h5()
        metricsgen.lgbm_save_feature_importance_plot()
        metricsgen.lgbm_classification_results()

        with open(pkl_model, "rb") as fin:
            gbm_model = pickle.load(fin)
        print(f"{target} Generating SHAP values...")
        explainer = shap.TreeExplainer(gbm_model)
        features_shap = features.sample(n=20000, random_state=seed, replace=False)
        shap_values = explainer.shap_values(features_shap)

        
        shap_expected=explainer.expected_value
        helpshap = shapHelpers(
            target, features_shap, shap_values, shap_expected, gbm_model, figfolder, datafolder, modelfolder) 
        helpshap.shap_save_to_disk()
        helpshap.shap_save_ordered_values()
        helpshap.save_requirements()
        helpshap.shap_prettify_column_names(
            prettycols_file=config.PRETTIFYING_COLUMNS_CSV
        )
        helpshap.shap_plot_summaries(
            title_in_figure=f"Impact of Variables on {name_for_figs} Prediction"
        )
        helpshap.shap_random_force_plots(
            n_plots=20, expected_value=shap_expected
        )
        helpshap.shap_top_dependence_plots(n_plots=10)
        helpshap.shap_top_dependence_plots_self(n_plots=20)
        helpshap.shap_int_vals_heatmap()

        print("This loop ran in:")
        print(datetime.now() - startTime1)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")