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

# Define readmission
thresh = config.READMISSION_THRESHOLD
data["readmitted"] = data["days_between_current_discharge_and_next_admission"] < thresh
data["readmitted"] = data["readmitted"] * 1  # convert to 1/0 rather than True/False

# Set index for splitting train/test/valid
data = data.sort_values(["admissiontime"]).reset_index(drop=False)
# data = data.set_index("admissiontime")

data["patientid"] = data["patientid"].astype("category")
data["admit_day_of_week"] = data["admit_day_of_week"].astype("category")
data["discharge_day_of_week"] = data["discharge_day_of_week"].astype("category")

data = data.drop(config.C_READMIT_DROP_COLS_LGBM, axis=1)

data_file = config.C_LGBM_FULL_DATA
data.to_pickle(data_file)

print("Building train and test...")

# Split the data by time of admission
data_train = data[config.TRAIN_START : config.TRAIN_END]
data_test = data[config.TEST_START : config.TEST_END]
data_valid = data[config.VALID_START : config.VALID_END]

# FOR DEBUGGING select small portion of dataset
# print("Selecting small portion for debugging...")
# data_train = data[0:15000]
# data_test = data[15000:19000]
# data_valid = data[19000:20000]

c_train_labels = data_train["readmitted"]
c_train_labels.to_pickle(config.C_TRAIN_LABELS_FILE)
c_train_features = data_train.drop(["readmitted"], axis=1)
c_train_features.to_pickle(config.C_TRAIN_FEATURES_FILE)

c_test_labels = data_test["readmitted"]
c_test_labels.to_pickle(config.C_TEST_LABELS_FILE)
c_test_features = data_test.drop(["readmitted"], axis=1)
c_test_features.to_pickle(config.C_TEST_FEATURES_FILE)

c_valid_labels = data_valid["readmitted"]
c_valid_labels.to_pickle(config.C_VALID_LABELS_FILE)
c_valid_features = data_valid.drop(["readmitted"], axis=1)
c_valid_features.to_pickle(config.C_VALID_FEATURES_FILE)

c_features = data.drop(["readmitted"], axis=1)
c_features.to_pickle(config.C_FEATURES_FILE)


# create lgb dataset files
print("Creating LightGBM datasets...")
c_d_train = lgb.Dataset(c_train_features, label=c_train_labels, free_raw_data=True)
c_d_test = lgb.Dataset(
    c_test_features, label=c_test_labels, reference=c_d_train, free_raw_data=True
)
c_d_valid = lgb.Dataset(
    c_valid_features, label=c_valid_labels, reference=c_d_train, free_raw_data=True
)

# remove old files if they exist
try:
    os.remove(config.LIGHTGBM_READMIT_TRAIN_00)
except OSError:
    pass

try:
    os.remove(config.LIGHTGBM_READMIT_TEST_00)
except OSError:
    pass

try:
    os.remove(config.LIGHTGBM_READMIT_VALID_00)
except OSError:
    pass


print("Saving to LightGBM binary files...")
c_d_train.save_binary(config.LIGHTGBM_READMIT_TRAIN_00)
c_d_test.save_binary(config.LIGHTGBM_READMIT_TEST_00)
c_d_valid.save_binary(config.LIGHTGBM_READMIT_VALID_00)

print("...complete.")


# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
