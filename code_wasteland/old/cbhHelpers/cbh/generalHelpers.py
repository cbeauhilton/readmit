import json
import os
import time

import h5py
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import config

jsonpickle_numpy.register_handlers()


def make_figfolder_for_target(debug, target):
    import time
    import os
    import config

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
    return figfolder


def make_modelfolder_for_target(debug, target):
    import time
    import os
    import config

    timestrfolder = time.strftime("%Y-%m-%d")
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
    return modelfolder


def make_datafolder_for_target(debug, target):
    import time
    import os
    import config

    timestrfolder = time.strftime("%Y-%m-%d")
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
    return datafolder


def make_report_tables_folder(debug):
    import os
    import config

    if debug:
        tablefolder = config.TABLES_DIR / "debug"
        if not os.path.exists(tablefolder):
            print("Making folder called", tablefolder)
            os.makedirs(tablefolder)
    else:
        tablefolder = config.TABLES_DIR
        if not os.path.exists(tablefolder):
            print("Making folder called", tablefolder)
            os.makedirs(tablefolder)
    return tablefolder


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikit's f1 doesn't like probabilities
    return "f1", f1_score(y_true, y_hat), True


def train_test_valid_80_10_10_split(data, target, seed):
    # Make sure there are no missing values in the target
    print("Data length before dropping null targets:", len(data))
    data = data[data[target].notnull()]
    if target == "length_of_stay_over_7_days":
        print("Dropping LoS outliers...")
        longies = len(data)
        data = data[data.length_of_stay_in_days < 40]
        data = data[data.length_of_stay_in_days > 0]
        shorties = len(data)
        print(f"Dropped {longies - shorties} LoS outliers.")

    # Split data to validation and train/test sets, 10% and 90%
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    for train_test_index, valid_index in split.split(data, data[target]):
        train_test_set = data.iloc[train_test_index]
        valid_set = data.iloc[valid_index]

    negclassnum = len(data[data[target] == 0])
    posclassnum = len(data[data[target] == 1])
    percentpos1 = posclassnum / len(data)
    print("Full set length", len(data), "(after dropping nulls, if any).")
    print("Number in negative class:", negclassnum)
    print("Number in positive class:", posclassnum)
    print("Percent in positive class is", percentpos1)
    print("\n")

    # Split train/test so the test is 10% of original and train is 80% of the original
    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - 8 / 9), random_state=seed
    )
    for train_index, test_index in split2.split(train_test_set, train_test_set[target]):
        train_set = train_test_set.iloc[train_index]
        test_set = train_test_set.iloc[test_index]

    negclassnum = len(train_test_set[train_test_set[target] == 0])
    posclassnum = len(train_test_set[train_test_set[target] == 1])
    percentpos = posclassnum / len(train_test_set)
    print("Train/test set length", len(train_test_set))
    print("Train/test set percent of total:", len(train_test_set) / len(data))
    print("Number in negative class:", negclassnum)
    print("Number in positive class:", posclassnum)
    print("Train/test percent in positive class is", percentpos)
    print("\n")

    negclassnum = len(train_set[train_set[target] == 0])
    posclassnum = len(train_set[train_set[target] == 1])
    percentpos = posclassnum / len(train_set)
    print("Train set length", len(train_set))
    print("Train set percent of total:", len(train_set) / len(data))
    print("Number in negative class:", negclassnum)
    print("Number in positive class:", posclassnum)
    print("Train percent in positive class is", percentpos)
    print("\n")

    negclassnum = len(test_set[test_set[target] == 0])
    posclassnum = len(test_set[test_set[target] == 1])
    percentpos = posclassnum / len(test_set)
    print("Test set length ", len(test_set))
    print("Test set percent of total:", len(test_set) / len(data))
    print("Number in negative class:", negclassnum)
    print("Number in positive class:", posclassnum)
    print("Test percent in positive class is ", percentpos)
    print("\n")

    negclassnum = len(valid_set[valid_set[target] == 0])
    posclassnum = len(valid_set[valid_set[target] == 1])
    percentpos = posclassnum / len(valid_set)
    print("Valid set length ", len(valid_set))
    print("Valid set percent of total:", len(valid_set) / len(data))
    print("Number in negative class:", negclassnum)
    print("Number in positive class:", posclassnum)
    print("Valid percent in positive class is ", percentpos)
    print("\n")

    # Note on formatting:
    # The colon allows you to specify options, the comma adds commas, the ".2f" says how many decimal points to keep.
    # Nice tutorial here: https://stackabuse.com/formatting-strings-with-python/
    if target == "readmitted30d":
        sentence01 = f"The cohort of hospitalizations was split into three groups for analysis: {len(train_set) / len(data) *100 :.0f}% for model development, "
        sentence02 = f"{len(test_set) / len(data) *100 :.0f}% for testing, and "
        sentence03 = f"{len(valid_set) / len(data) *100 :.0f}% for validation. "
        sentence03a = f"For example, the 30--day readmission cohort had development, testing, and validation cohorts of "
        sentence03b = f"n={len(train_set):,}, n={len(test_set):,}, and n={len(valid_set):,}, respectively. "
        sentence04 = f"Selection of hospitalizations for inclusion in each group was random with the exception "
        sentence05 = f"of ensuring the rate of the positive class (30-day readmission, length of stay over 5 days, etc.) was consistent between sets."
        paragraph01 = (
            sentence01
            + sentence02
            + sentence03
            + sentence03a
            + sentence03b
            + sentence04
            + sentence05
        )

        # Print paragraph to the terminal...
        print(paragraph01)
        print(f"Percent in positive class (whole set): {percentpos1*100 :.0f}%.")

        # Define file...
        if not os.path.exists(config.TEXT_DIR):
            print("Making folder called", config.TEXT_DIR)
            os.makedirs(config.TEXT_DIR)

        methods_text_file = config.TEXT_DIR / "methods_paragraphs.txt"

        # ...and save.
        with open(methods_text_file, "w") as text_file:
            print(paragraph01, file=text_file)

        text_file_latex = config.TEXT_DIR / "methods_paragraphs_latex.txt"
        # and make a LaTeX-friendly version (escape the % symbols with \)
        # Read in the file
        with open(methods_text_file, "r") as file:
            filedata = file.read()
        # Replace the target string
        filedata = filedata.replace("%", "\%")
        # Write the file
        with open(text_file_latex, "w") as file:
            file.write(filedata)
        print(f"LaTeX file saved to {text_file_latex}")

    return train_set, test_set, valid_set


def load_jsonified_sklearn_model_from_h5(filename, h5_model_key):
    with h5py.File(filename, "r") as f:
        print("Loading JSON model as clf...")
        h5_json_load = json.loads(
            f[h5_model_key][()]
        )  # takes a string and returns a dictionary
        h5_json_model = json.dumps(
            h5_json_load
        )  # takes a dictionary and returns a string
        clf = jsonpickle.decode(h5_json_model)  # requires a string, not dict
        print(clf)
        return clf


##############################################################################################################################

# print("Saving to file...")
# filename1 = config.CLEAN_PHASE_10
# data.to_pickle(filename1)
# print("Clean phase_0x available at:", filename1)
# print("\n")

# train_set_file = config.TRAIN_SET
# train_set.to_pickle(train_set_file)
# print("Train set available at:", train_set_file)
# print("\n")

# train_test_set_file = config.TRAIN_TEST_SET
# train_test_set.to_pickle(train_test_set_file)
# print("Train_test set available at:", train_test_set_file)
# print("\n")

# under_sample_file = config.UNDERSAMPLE_TRAIN_SET
# under_sample_train.to_pickle(under_sample_file)
# print("Undersample train/test set available at:", under_sample_file)
# print("\n")

# test_set_file = config.TEST_SET
# test_set.to_pickle(test_set_file)
# print("Test set available at:", test_set_file)
# print("\n")

# valid_set_file = config.VALID_SET
# valid_set.to_pickle(valid_set_file)
# print("Valid set available at:", valid_set_file)
# print("\n")

##############################################################################################################################
