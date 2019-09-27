import sys

sys.path.append("modules")

from datetime import datetime

startTime = datetime.now()

import os
import pandas as pd
import lightgbm as lgb

import config

pd.options.display.max_columns = 2000

# Load file
filename = config.PROCESSED_FINAL
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")


data["patientid"] = data["patientid"].astype("category")
data["admit_day_of_week"] = data["admit_day_of_week"].astype("category")
# data["discharge_day_of_week"] = data["discharge_day_of_week"].astype("category")
data["primary_language"] = data["primary_language"].astype("category")

print("Building train and test...")

# Split the data by time of admission

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

# FOR DEBUGGING
# Select random sample of dataset:
print("Selecting small portion for debugging...")
data_train = data_train.sample(frac=0.0005, random_state=config.SEED)
data_test = data_test.sample(frac=0.0005, random_state=config.SEED)
data_valid = data_valid.sample(frac=0.0005, random_state=config.SEED)
data = data.sample(frac=0.0005, random_state=config.SEED)

data_train = data_train.drop(config.R_LOS_DROP_COLS_LGBM, axis=1)
data_test = data_test.drop(config.R_LOS_DROP_COLS_LGBM, axis=1)
data_valid = data_valid.drop(config.R_LOS_DROP_COLS_LGBM, axis=1)
data = data.drop(config.R_LOS_DROP_COLS_LGBM, axis=1)
data_file = config.R_LOS_LGBM_FULL_DATA
data.to_pickle(data_file)
print("All the features in the final set:", list(data))

r_train_labels = data_train["length_of_stay_in_days"]
r_train_labels.to_pickle(config.R_TRAIN_LABELS_FILE)
r_train_features = data_train.drop(["length_of_stay_in_days"], axis=1)
r_train_features.to_pickle(config.R_TRAIN_FEATURES_FILE)

r_test_labels = data_test["length_of_stay_in_days"]
r_test_labels.to_pickle(config.R_TEST_LABELS_FILE)
r_test_features = data_test.drop(["length_of_stay_in_days"], axis=1)
r_test_features.to_pickle(config.R_TEST_FEATURES_FILE)

r_valid_labels = data_valid["length_of_stay_in_days"]
r_valid_labels.to_pickle(config.R_VALID_LABELS_FILE)
r_valid_features = data_valid.drop(["length_of_stay_in_days"], axis=1)
r_valid_features.to_pickle(config.R_VALID_FEATURES_FILE)

r_features = data.drop(["length_of_stay_in_days"], axis=1)
r_features.to_pickle(config.R_FEATURES_FILE)


# create lgb dataset files
print("Creating LightGBM datasets...")
r_d_train = lgb.Dataset(r_train_features, label=r_train_labels, free_raw_data=True)
r_d_test = lgb.Dataset(
    r_test_features, label=r_test_labels, reference=r_d_train, free_raw_data=True
)
r_d_valid = lgb.Dataset(
    r_valid_features, label=r_valid_labels, reference=r_d_train, free_raw_data=True
)

# remove old files if they exist
try:
    os.remove(config.R_LIGHTGBM_READMIT_TRAIN_00)
except OSError:
    pass

try:
    os.remove(config.R_LIGHTGBM_READMIT_TEST_00)
except OSError:
    pass

try:
    os.remove(config.R_LIGHTGBM_READMIT_VALID_00)
except OSError:
    pass


print("Saving to LightGBM binary files...")
r_d_train.save_binary(config.R_LIGHTGBM_READMIT_TRAIN_00)
r_d_test.save_binary(config.R_LIGHTGBM_READMIT_TEST_00)
r_d_valid.save_binary(config.R_LIGHTGBM_READMIT_VALID_00)

print("...complete.")


# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
