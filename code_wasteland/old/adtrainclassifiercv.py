import sys

sys.path.append("modules")
from datetime import datetime

startTime = datetime.now()

import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit

try:
    import cPickle as pickle
except BaseException:
    import pickle
import config

# import configdocker as config

seed = config.SEED

# Load file
filename = config.PROCESSED_FINAL
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")

### 30d readmission is already in the set - if you want to 
### try another threshold, add it here and drop the 30d column

# (Re)define readmission
# thresh = config.READMISSION_THRESHOLD
# data["readmitted"] = data["days_between_current_discharge_and_next_admission"] < thresh
# data["readmitted"] = data["readmitted"] * 1  # convert to 1/0 rather than True/False

# data["readmitted"]  = data["readmitted30d"]
data["readmitted"]  = data["readmittedpast30d"]

data = data.drop(["readmitted30d"],axis=1)
data = data.drop(["readmittedpast30d"],axis=1)

# Set index for splitting train/test/valid
# data = data.sort_values(["admissiontime"]).reset_index(drop=False)
# data = data.set_index("admissiontime")

# Set index for splitting train/test/valid
# data = data.sort_values(["admissiontime"]).reset_index(drop=False)
# data = data.set_index("admissiontime")

data["patientid"] = data["patientid"].astype("category")
data["admit_day_of_week"] = data["admit_day_of_week"].astype("category")
data["discharge_day_of_week"] = data["discharge_day_of_week"].astype("category")
data["primary_language"] = data["primary_language"].astype("category")

print("Building train and valid for CV...")


data_cv = data[
    (data["admissiontime"] > config.TRAIN_START)
    & (data["admissiontime"] < config.TEST_END)
]

data_valid = data[
    (data["admissiontime"] > config.VALID_START)
    & (data["admissiontime"] < config.VALID_END)
]

# FOR DEBUGGING
# Select random sample of dataset:
print("Selecting small portion for debugging...")
data_cv = data_cv.sample(frac=0.0005, random_state=config.SEED)
data_valid = data_valid.sample(frac=0.0005, random_state=config.SEED)
data = data.sample(frac=0.0005, random_state=config.SEED)

data_cv = data_cv.drop(config.C_READMIT_DROP_COLS_LGBM, axis=1)
data_valid = data_valid.drop(config.C_READMIT_DROP_COLS_LGBM, axis=1)
data = data.drop(config.C_READMIT_DROP_COLS_LGBM, axis=1)
data_file = config.C_LGBM_FULL_DATA_CV
data.to_pickle(data_file)

c_train_labels = data_cv["readmitted"]
c_train_labels.to_pickle(config.C_TRAIN_LABELS_FILE_CV)
c_train_features = data_cv.drop(["readmitted"], axis=1)
c_train_features.to_pickle(config.C_TRAIN_FEATURES_FILE_CV)
c_valid_labels = data_valid["readmitted"]
c_valid_labels.to_pickle(config.C_VALID_LABELS_FILE_CV)
c_valid_features = data_valid.drop(["readmitted"], axis=1)
c_valid_features.to_pickle(config.C_VALID_FEATURES_FILE_CV)
c_features = data.drop(["readmitted"], axis=1)
c_features.to_pickle(config.C_FEATURES_FILE_CV)

folds = TimeSeriesSplit(n_splits=5)
folds_generator = folds.split(c_train_features, c_train_labels)
# print(tscv)  
# for train_index, test_index in tscv.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     c_d_train, c_d_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

# create lgb dataset files
print("Creating LightGBM datasets...")
c_d_train = lgb.Dataset(c_train_features, label=c_train_labels, free_raw_data=True)
c_d_valid = lgb.Dataset(
    c_valid_features, label=c_valid_labels, reference=c_d_train, free_raw_data=True
)


print("Training...")
c_evals_result = {}  # to record eval results for plotting

c_params = config.C_READMIT_PARAMS_LGBM

c_gbm = lgb.cv(
    c_params,
    c_d_train,
    num_boost_round=100000,
    early_stopping_rounds=100,
    verbose_eval=25,
    # evals_result=c_evals_result,
    # keep_training_booster=True,
    nfold=5,
)

print("Dumping model with pickle...")
pkl_model = config.LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE_CV
with open(pkl_model, "wb") as fout:
    pickle.dump(c_gbm, fout)

# To continue training...
# print("Moar training...")
# c_gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10,
#                 init_model=pkl_model,
#                 valid_sets=c_d_test)

# c_gbm.save_model(config.LGBM_READMIT_MODEL_CLASSIFICATION)
# print("Dumping to pickle...")
# with open(pkl_model, "wb") as fout:
#     pickle.dump(c_gbm, fout)
# print("Done.")

# predict probabilities
# c_predict_labels = c_gbm.predict(c_test_features)

# fpr, tpr, threshold = metrics.roc_curve(c_test_labels, c_predict_labels)
# roc_auc = metrics.auc(fpr, tpr)

# print("Plot metrics recorded during training...")
# ax = lgb.plot_metric(c_evals_result, metric="auc", figsize=(10, 10))
# figure_title = "CV_Max AUC %0.2f related to iterations_" % roc_auc
# timestr = time.strftime("%Y-%m-%d-%H%M")
# ext = ".png"
# title = figure_title + timestr + ext
# plt.savefig(
#     (config.FIGURES_DIR / title), dpi=400, transparent=False, bbox_inches="tight"
# )
# # plt.show()
# plt.close()

# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
