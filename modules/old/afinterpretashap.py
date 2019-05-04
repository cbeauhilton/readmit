import sys

sys.path.append("modules")


from datetime import datetime
import time
from pathlib import Path
startTime = datetime.now()

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import random
import numpy as np
import scipy
from imblearn.metrics import classification_report_imbalanced
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

try:
    import cPickle as pickle
except BaseException:
    import pickle

import config

seed = config.SEED

print("Loading model and SHAP values from pickle...")

pkl_model = Path(r"C:\Users\hiltonc\Desktop\readmit\models\2019-02-27\readmitted30d MODEL 2019-02-27-2225.pickle")
pkl_shap = Path(r"C:\Users\hiltonc\Desktop\readmit\models\2019-02-28\readmitted30d SHAP 2019-02-28-0622.pickle")
features_file = Path(r"C:\Users\hiltonc\Desktop\readmit\data\processed\2019-02-28\readmitted30d c_features 2019-02-28-0738.pickle")  
test_features_file = Path(r"C:\Users\hiltonc\Desktop\readmit\data\processed\2019-02-28\readmitted30d c_test_features 2019-02-28-0738.pickle")  
test_labels_file  = Path(r"C:\Users\hiltonc\Desktop\readmit\data\processed\2019-02-28\readmitted30d c_test_labels 2019-02-28-0738.pickle")  


# load model with pickle to predict
c_test_features = pd.read_pickle(test_features_file)
c_test_labels = pd.read_pickle(test_labels_file)
c_features = pd.read_pickle(features_file)

with open(pkl_model, "rb") as fin:
    c_gbm = pickle.load(fin)

explainer = shap.TreeExplainer(c_gbm)

with open(pkl_shap, "rb") as f:
    shap_values = pickle.load(f)

# shap_values = pd.read_pickle(config.C_LGBM_SHAP_FILE)
print("Making SHAP summary bar plot...")
shap.summary_plot(
    shap_values,
    c_features,
    title="Impact of Variables on Readmission Prediction",
    plot_type="bar",
    show=False,
)

figure_title = "30BREADMIT_SHAP_value_summary_bar_plot_classification_"
timestr = time.strftime("%Y-%m-%d-%H%M")
ext = ".png"
title = figure_title + timestr + ext
plt.savefig(
    (config.FIGURES_DIR / title), dpi=400, transparent=False, bbox_inches="tight"
)
plt.close()

print("Making SHAP summary plot...")
shap.summary_plot(
    shap_values,
    c_features,
    title="Impact of Variables on Readmission Prediction",
    show=False,
)

figure_title = "30BREADMIT_SHAP_value_summary_plot_classification_"
timestr = time.strftime("%Y-%m-%d-%H%M")
ext = ".png"
title = figure_title + timestr + ext
plt.savefig(
    (config.FIGURES_DIR / title), dpi=400, transparent=False, bbox_inches="tight"
)
plt.close()


shap_indices = np.random.choice(shap_values.shape[0], 5)  # select 5 random patients
# shap_probs = scipy.special.expit(shap_values)  #+ explainer.expected_value
# shap_indices = np.random.choice(shap_probs.shape[0], 5)  # select 5 random patients

for pt_num in shap_indices:
    # shap.force_plot(
    #     explainer.expected_value, # sets baseline value, based on balance of positive and negative samples
    #     shap_values[pt_num, :], # grabs components of prediction for a given sample
    #     c_features.iloc[pt_num, :], # grabs the identities and values of components
    #     text_rotation=30, # easier to read
    #     matplotlib=True, # instead of JS
    #     show=False, # allows saving, etc.
    # )
    print("Making force plot for patient ", pt_num, "...")
    shap.force_plot(
        0, # set expected value to zero
        np.hstack([shap_values[pt_num, :], explainer.expected_value]), # add expected value as a bar in the force plot
        pd.concat(
            [c_features, pd.DataFrame({"Base value": [explainer.expected_value]})],
            axis=1,
        ).iloc[pt_num, :], # grabs the identities and values of components
        text_rotation=30, # easier to read
        matplotlib=True, # instead of Javascript
        show=False, # allows saving, etc.
        link="logit",
    )

    figure_title = "30BREADMIT_SHAP Force Plot "
    patient_number = f" Pt {pt_num}"
    timestr = time.strftime("%Y-%m-%d-%H%M")
    ext = ".png"
    title = figure_title + timestr + patient_number + ext
    plt.savefig(
        (config.FORCE_PLOT_SINGLETS / title),
        dpi=400,
        transparent=False,
        bbox_inches="tight",
    )
    plt.close()


# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
