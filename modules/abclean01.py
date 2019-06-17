import sys

sys.path.append("modules")

from datetime import datetime



import os
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import numpy as np
import time

tqdm.pandas()

from cbh import config

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

pd.options.display.max_columns = 2000

# Load file
filename = config.CLEAN_PHASE_00
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")

print("Condensing gender category...")
data = data[
    (data.gender == "Female") | (data.gender == "Male")
]  # Judith Butler is crying

print("Condensing primary_language category...")
data.loc[
    data.groupby("primary_language").primary_language.transform("count").lt(1500),
    "primary_language",
] = "Language.other"  # make all languages with <1500 speakers into one category
d = {
    "Language not recorded": "Language_other",
    "Language.other": "Language_other",
}  # lump the unrecorded and rare languages
data.primary_language = data.primary_language.replace(d)

death_date_cols = ["epicdeathdate", "ohiodeathindexdate", "socialsecuritydeathdate"]

for col in death_date_cols:
    data[col] = pd.to_datetime(data[col])
    # data[col] = pd.to_datetime(data[col], yearfirst=True)
    timecolname = f"time_from_admission_to_{col}"
    timecolname_2 = f"time_from_discharge_to_{col}"
    print(timecolname)
    data[timecolname] = data[col] - data["admissiontime"]
    data[timecolname_2] = data[col] - data["dischargetime"]
    hourscolname = f"hours_from_admission_to_{col}"
    print(hourscolname)
    data[hourscolname] = data.progress_apply(
        lambda row: row[timecolname].total_seconds(), axis=1
    )
    data[hourscolname] = (
        data[hourscolname] / 3600
    )  # divide seconds by 3600 to get hours

    # get rid of patients with death dates that are somehow before the admissions
    # but keep the NaNs (most of the data)
    # 1633367 - 1629981 = 3386 pts dropped by this operation
    data = data[(data[hourscolname] >= 0) | (data[hourscolname].isnull())]
    print(f"died_within_48_72h_of_admission_{col}")
    data[f"died_within_48_72h_of_admission_{col}"] = (
        data[hourscolname].between(48, 72, inclusive=True) * 1.0
    )

    # died at _any time_ during this admission
    colname = f"died_in_this_admission_{col}"
    print(colname)
    data[colname] = (
        data.progress_apply(
            lambda row: (row.admissiontime <= row[col] <= row.dischargetime), axis=1
        )
        * 1.0
    )


# if there is a 1 in any of the "died in this admission" columns,
# make a 1 in a new combo column. The “|” means “or”.
conditions = [
    (data["died_in_this_admission_epicdeathdate"] == 1)
    | (data["died_in_this_admission_ohiodeathindexdate"] == 1)
    | (data["died_in_this_admission_socialsecuritydeathdate"] == 1)
]
choices = [1]
data["died_in_this_admission"] = np.select(conditions, choices, default=0)

print("Combining death info from Epic, Ohio, and Social Security...")
data["died_within_48_72h_of_admission_combined"] = data.progress_apply(
    lambda row: row.died_within_48_72h_of_admission_epicdeathdate
    + row.died_within_48_72h_of_admission_ohiodeathindexdate
    + row.died_within_48_72h_of_admission_socialsecuritydeathdate,
    axis=1,
)  # add booleans, should sum to 3 at max

data["died_within_48_72h_of_admission_combined"] = (
    data["died_within_48_72h_of_admission_combined"].between(1, 111, inclusive=True)
    * 1.0
)  # select patients with at least one recorded death date


# Save features to csv
import time
timestrfolder = time.strftime("%Y-%m-%d")
datafolder = config.PROCESSED_DATA_DIR / timestrfolder
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
feature_list = list(data)
df = pd.DataFrame(feature_list, columns=["features"])
spreadsheet_title = "Feature list 01 "
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".csv"
title = spreadsheet_title + timestr + ext
feature_list_file = datafolder / title
df.to_csv(feature_list_file, index=False)
print("CSV of features available at: ", feature_list_file)


# print(list(data))

print("Saving to file...")
filename1 = config.CLEAN_PHASE_01
data.to_pickle(filename1)
print("Clean phase_01 available at:", filename1)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
