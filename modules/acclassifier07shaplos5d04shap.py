import glob
import os
import time
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.exceptions import UndefinedMetricWarning

import cbh.config as config
import cbh.configcols as configcols
from cbh.generalHelpers import (lgb_f1_score, make_datafolder_for_target,
                                make_figfolder_for_target,
                                make_modelfolder_for_target,
                                make_report_tables_folder,
                                train_test_valid_80_10_10_split)
from cbh.lgbmHelpers import lgbmClassificationHelpers
from cbh.shapHelpers import shapHelpers

try:
    import cPickle as pickle
except BaseException:
    import pickle

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

seed = config.SEED
debug = False

target = config.TARGET 
name_for_figs = config.NAME_FOR_FIGS

class_thresh = 0.5

print("Debug:", debug)

final_file = config.PROCESSED_DATA_DIR / f"{target}.h5"
data = pd.read_hdf(final_file, key=f"{target}clean")
if debug:
    data = data[:20000]

print("File loaded.")

figfolder = make_figfolder_for_target(debug, target)
datafolder = make_datafolder_for_target(debug, target)
modelfolder = make_modelfolder_for_target(debug, target)
tablefolder = make_report_tables_folder(debug)

features = data.drop([target], axis=1)
features = features.drop(["length_of_stay_in_days"], axis=1)


print("Loading model...")
# find most recently modified pickle file in the model folder
pkl_model = max(glob.iglob(os.path.join(modelfolder, '*MODEL*.pickle')), key=os.path.getmtime)

# from datetime import datetime
# last_modified = datetime.fromtimestamp(os.path.getmtime(pkl_model)).strftime("%H%M")
# print(f"last modified: {last_modified}")

with open(pkl_model, "rb") as f:
    gbm_model = pickle.load(f)

print(f"{target} Generating SHAP values...")
explainer = shap.TreeExplainer(gbm_model)
features_shap = features.sample(n=20000, random_state=seed, replace=False)
shap_values = explainer.shap_values(features_shap)

shap_expected = explainer.expected_value
print(shap_expected)
helpshap = shapHelpers(
    target,
    name_for_figs,
    class_thresh,
    features_shap,
    shap_values,
    shap_expected,
    gbm_model,
    figfolder,
    datafolder,
    modelfolder,
)
helpshap.shap_save_to_disk()
helpshap.shap_save_ordered_values()
helpshap.save_requirements()
helpshap.shap_prettify_column_names(prettycols_file=config.PRETTIFYING_COLUMNS_CSV)
helpshap.shap_plot_summaries(
    title_in_figure=f"Impact of Variables on {name_for_figs} Prediction"
)
helpshap.shap_random_force_plots(n_plots=20, expected_value=shap_expected)
helpshap.shap_top_dependence_plots(n_plots=10)
helpshap.shap_top_dependence_plots_self(n_plots=20)
helpshap.shap_int_vals_heatmap()

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")

# LOS 5d expected: [1.124068978234447, -1.124068978234447]
