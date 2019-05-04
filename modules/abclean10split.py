import sys

sys.path.append("modules")

from datetime import datetime

import os
import pandas as pd
import lightgbm as lgb
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedShuffleSplit

import config
from zz_generalHelpers import *

seed = config.SEED
pd.options.display.max_columns = 2000

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

# Load file
filename = config.CLEAN_PHASE_09
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")

print("Dropping outliers wrt age and LoS...")
print("Data length before dropping LoS outliers :", len(data))
data = data[data.length_of_stay_in_days < 40]
data = data[data.length_of_stay_in_days > 0]
print(
    "Data length after dropping LoS outliers and before dropping patient age outliers :",
    len(data),
)
# data = data[data.patient_age < 130]  # https://en.wikipedia.org/wiki/Oldest_people
# data = data[
#     data.patient_age >= 0
# ]  # negative ages snuck in somehow. Using (>= 0) allows newborns.
# print("Data length after dropping patient age outliers :", len(data))
# data = data[data.days_between_current_discharge_and_next_admission > 1]
# data = data[data.days_between_current_admission_and_previous_discharge > 1]
# print("Data length after dropping probable CCF transfers :", len(data))

# Split data to validation and train/test sets, 10% and 90%
split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
for train_test_index, valid_index in split.split(data, data.readmitted30d):
    train_test_set = data.iloc[train_test_index]
    valid_set = data.iloc[valid_index]

notreadmittednum = len(data[data["readmitted30d"] == 0])
readmittednum = len(data[data["readmitted30d"] == 1])
totalnum = len(data)
percentreadmitted = readmittednum / totalnum
print("Full set length ", len(data), "encounters.")
print(notreadmittednum)
print(readmittednum)
print("Percent readmitted is ", percentreadmitted)
print("\n")

# Split train/test so the test is 10% of original and train is 80% of the original
split2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - 8 / 9), random_state=seed)
for train_index, test_index in split2.split(
    train_test_set, train_test_set.readmitted30d
):
    train_set = train_test_set.iloc[train_index]
    test_set = train_test_set.iloc[test_index]

num_readmits = len(train_set[train_set["readmitted30d"] == 1])
non_readmit_indices = train_set[train_set.readmitted30d == 0].index
random_indices = np.random.choice(non_readmit_indices, num_readmits, replace=False)
readmit_indices = train_set[train_set.readmitted30d == 1].index
under_sample_indices = np.concatenate([readmit_indices, random_indices])
under_sample_train = train_set.loc[under_sample_indices]

notreadmittednum = len(train_test_set[train_test_set["readmitted30d"] == 0])
readmittednum = len(train_test_set[train_test_set["readmitted30d"] == 1])
totalnum = len(train_test_set)
percentreadmitted = readmittednum / totalnum
print("Train/test set length ", len(train_test_set))
print("Train/test set percent of total:", len(train_test_set) / len(data))
print(notreadmittednum)
print(readmittednum)
print("Train/test percent readmitted is ", percentreadmitted)
print("\n")

notreadmittednum = len(train_set[train_set["readmitted30d"] == 0])
readmittednum = len(train_set[train_set["readmitted30d"] == 1])
totalnum = len(train_set)
percentreadmitted = readmittednum / totalnum
print("Train set length ", len(train_set))
print("Train set percent of total:", len(train_set) / len(data))
print(notreadmittednum)
print(readmittednum)
print("Train percent readmitted is ", percentreadmitted)
print("\n")

notreadmittednum = len(under_sample_train[under_sample_train["readmitted30d"] == 0])
readmittednum = len(under_sample_train[under_sample_train["readmitted30d"] == 1])
percentreadmitted = readmittednum / notreadmittednum
print("Undersample train set length ", len(under_sample_train))
print("Undersample train set percent of total:", len(under_sample_train) / len(data))
print(notreadmittednum)
print(readmittednum)
print("Undersample train percent readmitted is ", percentreadmitted)
print("\n")

notreadmittednum = len(test_set[test_set["readmitted30d"] == 0])
readmittednum = len(test_set[test_set["readmitted30d"] == 1])
totalnum = len(test_set)
percentreadmitted = readmittednum / totalnum
print("Test set length ", len(test_set))
print("Test set percent of total:", len(test_set) / len(data))
print(notreadmittednum)
print(readmittednum)
print("Test percent readmitted is ", percentreadmitted)
print("\n")

notreadmittednum = len(valid_set[valid_set["readmitted30d"] == 0])
readmittednum = len(valid_set[valid_set["readmitted30d"] == 1])
totalnum = len(valid_set)
percentreadmitted = readmittednum / totalnum
print("Valid set length ", len(valid_set))
print("Valid set percent of total:", len(valid_set) / len(data))
print(notreadmittednum)
print(readmittednum)
print("Valid percent readmitted is ", percentreadmitted)
print("\n")


print("Saving to file...")
filename1 = config.CLEAN_PHASE_10
data.to_pickle(filename1)
print("Clean phase_0x available at:", filename1)
print("\n")

train_set_file = config.TRAIN_SET
train_set.to_pickle(train_set_file)
print("Train set available at:", train_set_file)
print("\n")

train_test_set_file = config.TRAIN_TEST_SET
train_test_set.to_pickle(train_test_set_file)
print("Train_test set available at:", train_test_set_file)
print("\n")

under_sample_file = config.UNDERSAMPLE_TRAIN_SET
under_sample_train.to_pickle(under_sample_file)
print("Undersample train/test set available at:", under_sample_file)
print("\n")

test_set_file = config.TEST_SET
test_set.to_pickle(test_set_file)
print("Test set available at:", test_set_file)
print("\n")

valid_set_file = config.VALID_SET
valid_set.to_pickle(valid_set_file)
print("Valid set available at:", valid_set_file)
print("\n")

# Note on formatting:
# The colon allows you to specify options, the comma adds commas, the ".2f" says how many decimal points to keep.
# Nice tutorial here: https://stackabuse.com/formatting-strings-with-python/ 

sentence01 = f"The cohort of hospitalizations was split into three groups for analysis: {len(train_set) / len(data) *100 :.0f}% for model development (n={len(train_set):,}), "
sentence02 = f"{len(test_set) / len(data) *100 :.0f}% for testing (n={len(test_set):,}), and "
sentence03 = f"{len(valid_set) / len(data) *100 :.0f}% for validation (n={len(valid_set):,}). " 
sentence04 = f"Selection of hospitalizations for inclusion in each group was random with the exception of ensuring the readmission rate was consistently {percentreadmitted*100 :.0f}%." 
paragraph01 = sentence01+sentence02+sentence03+sentence04

# Print paragraph to the terminal...
print(paragraph01)

# Define file...
if not os.path.exists(config.TEXT_DIR):
    print("Making folder called", config.TEXT_DIR)
    os.makedirs(config.TEXT_DIR)

methods_text_file = config.TEXT_DIR/"methods_paragraphs.txt"

#...and save.
with open(methods_text_file, "w") as text_file:
    print(paragraph01, file=text_file)


text_file_latex = config.TEXT_DIR / "methods_paragraphs_latex.txt"
# and make a LaTeX-friendly version (escape the % symbols with \)
# Read in the file
with open(methods_text_file, 'r') as file:
  filedata = file.read()
# Replace the target string
filedata = filedata.replace('%', '\%')
# Write the file
with open(text_file_latex, 'w') as file:
  file.write(filedata)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")


