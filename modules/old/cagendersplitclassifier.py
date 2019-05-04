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

### 30d readmission is already in the set - if you want to 
### try another threshold, add it here and drop the 30d column

# Define readmission
# data = data.drop("readmitted30d", axis=1)
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
data["gender"] = data["gender"].replace(['Female'], 1)
data["gender"] = data["gender"].replace(['Male'], 0)
# data["gender"] = data["gender"].replace(["Gender unknown"], "NaN")
# data["gender"] = data.gender.notnull()
data =data[(data.gender == 0) | (data.gender == 1)]
data["gender"] = data["gender"].astype(int)
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

data_train = data_train.drop(config.C_READMIT_DROP_COLS_LGBM, axis=1)
data_test = data_test.drop(config.C_READMIT_DROP_COLS_LGBM, axis=1)
data_valid = data_valid.drop(config.C_READMIT_DROP_COLS_LGBM, axis=1)

data = data.drop(config.C_READMIT_DROP_COLS_LGBM, axis=1)
data_file = config.C_LGBM_FULL_DATA
data.to_pickle(data_file)

# FOR DEBUGGING
# Select random sample of dataset:
# print("Selecting small portion for debugging...")
# data_train = data_train.sample(frac=0.0005, random_state=config.SEED)
# data_test = data_test.sample(frac=0.0005, random_state=config.SEED)
# data_valid = data_valid.sample(frac=0.0005, random_state=config.SEED)
# data = data.sample(frac=0.0005, random_state=config.SEED)

c_gender_train_labels = data_train["gender"]
c_gender_train_labels.to_pickle(config.C_GENDER_TRAIN_LABELS_FILE)
c_gender_train_features = data_train.drop(["gender"], axis=1)
c_gender_train_features.to_pickle(config.C_GENDER_TRAIN_FEATURES_FILE)

c_gender_test_labels = data_test["gender"]
c_gender_test_labels.to_pickle(config.C_GENDER_TEST_LABELS_FILE)
c_gender_test_features = data_test.drop(["gender"], axis=1)
c_gender_test_features.to_pickle(config.C_GENDER_TEST_FEATURES_FILE)

c_gender_valid_labels = data_valid["gender"]
c_gender_valid_labels.to_pickle(config.C_GENDER_VALID_LABELS_FILE)
c_gender_valid_features = data_valid.drop(["gender"], axis=1)
c_gender_valid_features.to_pickle(config.C_GENDER_VALID_FEATURES_FILE)

c_gender_features = data.drop(["gender"], axis=1)
c_gender_features.to_pickle(config.C_GENDER_FEATURES_FILE)


# create lgb dataset files
print("Creating LightGBM datasets...")
c_gender_d_train = lgb.Dataset(c_gender_train_features, label=c_gender_train_labels, free_raw_data=True)
c_gender_d_test = lgb.Dataset(
    c_gender_test_features, label=c_gender_test_labels, reference=c_gender_d_train, free_raw_data=True
)
c_gender_d_valid = lgb.Dataset(
    c_gender_valid_features, label=c_gender_valid_labels, reference=c_gender_d_train, free_raw_data=True
)

# remove old files if they exist
try:
    os.remove(config.LIGHTGBM_GENDER_READMIT_TRAIN_00)
except OSError:
    pass

try:
    os.remove(config.LIGHTGBM_GENDER_READMIT_TEST_00)
except OSError:
    pass

try:
    os.remove(config.LIGHTGBM_GENDER_READMIT_VALID_00)
except OSError:
    pass


print("Saving to LightGBM binary files...")
c_gender_d_train.save_binary(config.LIGHTGBM_GENDER_READMIT_TRAIN_00)
c_gender_d_test.save_binary(config.LIGHTGBM_GENDER_READMIT_TEST_00)
c_gender_d_valid.save_binary(config.LIGHTGBM_GENDER_READMIT_VALID_00)

print("...complete.")


# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
