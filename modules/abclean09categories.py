import json
import os
import sys
import zipfile
from datetime import datetime
from io import StringIO
from urllib.request import urlopen

from pathlib import Path
import numpy as np

import censusdata
import pandas as pd
from pandas.io.json import json_normalize
from tqdm import tqdm

tqdm.pandas()

sys.path.append("modules")
from cbh import config
import configcols

print("About to run", os.path.basename(__file__))

startTime = datetime.now()

interim_dir = config.INTERIM_DATA_DIR

data_file = config.CLEAN_PHASE_04
print("Loading", data_file)
result = pd.read_pickle(data_file)

before_dedup = len(result)
result = result.drop_duplicates(subset=["encounterid"], keep="first")
after_dedup = len(result)
print(f"Dropped {before_dedup - after_dedup} duplicate encounters")

result = result[result.patient_age < 130]  # https://en.wikipedia.org/wiki/Oldest_people
result = result[result.patient_age >= 0]
agism = len(result)
# negative ages snuck in somehow. Using (>= 0) allows newborns.
print(f"Dropped {after_dedup - agism} impossibly aged encounters")

# Define readmission at various thresholds
readmission_thresholds = [3, 5, 7, 15, 20, 28, 30, 45, 90, 180, 365, 3650]
for thresh in readmission_thresholds:
    print(f"Making column for readmission threshold at {thresh} days...")
    result[f"readmitted{thresh}d"] = (
        result["days_between_current_discharge_and_next_admission"] <= thresh
    ) & (
        result["days_between_current_discharge_and_next_admission"] > 0.15
    )  # adding 4 hours accounts for transfers, based on histogram analysis
    result[f"readmitted{thresh}d"] = (
        result[f"readmitted{thresh}d"] * 1
    )  # convert to 1/0 rather than True/False

# Mobilize some of the ACS data
result["over_200_ratio"] = (
    result["acs_200_and_over_ratio_income_poverty_level_past_12_mo"]
    / result["acs_total_population_count"]
)
result["white_alone_ratio"] = (
    result["acs_race_white_alone"] / result["acs_total_population_count"]
)

# BMI delta
result["BMI_delta"] = pd.to_numeric(
    result["bmi_discharge"], errors="coerce"
) - pd.to_numeric(result["bmi_admit"], errors="coerce")
# print(result["BMI_delta"])

# BP deltas
result["discharge_diastolic_delta"] = pd.to_numeric(
    result["discharge_diastolic_bp"], errors="coerce"
) - pd.to_numeric(result["admit_diastolic_bp"], errors="coerce")
result["discharge_systolic_delta"] = pd.to_numeric(
    result["discharge_systolic_bp"], errors="coerce"
) - pd.to_numeric(result["admit_systolic_bp"], errors="coerce")
# print(result.discharge_diastolic_delta)

print("Binarizing patient age...")
age_thresholds = [10, 30, 65]
for thresh in age_thresholds:
    print(f"Making binary column for age at {thresh} years...")
    result[f"age_gt_{thresh}_y"] = result["patient_age"] >= thresh
    result[f"age_gt_{thresh}_y"] = result[f"age_gt_{thresh}_y"] * 1
    # convert to 1/0 rather than True/False

print("Converting diagnosis codes to diagnosis descriptions as available...")
dxcode_file = config.DX_CODES_CONVERTED
dxcodes = pd.read_pickle(dxcode_file)
di = pd.Series(
    dxcodes["diagnosis description"].values, index=dxcodes["diagnosis code"]
).to_dict()

# result["primary_diagnosis_code"].replace(di, inplace=True) # super slow - use map instead
# https://stackoverflow.com/questions/20250771/remap-values-in-pandas-column-with-a-dict

result["primary_diagnosis_code"] = (
    result["primary_diagnosis_code"].map(di).fillna(result["primary_diagnosis_code"])
)

print("Dropping negative LoS rows (n=97, all same-day)...")
result = result[result["length_of_stay_in_days"] >= 0]

print(
    "Dropping encounters with a discharge disposition of Expired and somehow were also readmitted within 30d (n=43)"
)
print("Before", len(result))
result = result.drop(
    result[
        (
            (result.dischargedispositiondescription == "Expired")
            & (result.readmitted30d > 0)
        )
    ].index
)
print("After", len(result))

datecols = result.filter(items=configcols.DATETIME_COLS)

print("Dropping datetime cols...")
result = result.drop(configcols.DATETIME_COLS, axis=1, errors="ignore")

# fix values wrt casing, bad spacing, etc.
print("Cleaning text within cells...")
result = result.progress_apply(
    lambda x: x.str.strip()
    .str.replace("\t", "")
    .str.replace("_", " ")
    .str.replace("__", " ")
    .str.replace("-", " ")
    .str.replace(", ", " ")
    .str.replace(",", " ")
    .str.replace("/", "")
    .str.replace("'", "")
    .str.capitalize()
    if (x.dtype == "object")
    else x
)

result = pd.merge(result, datecols, left_index=True, right_index=True)

print("Setting categorical cols astype _category_...")
cat_cols = configcols.CATEGORICAL_COLS
result[cat_cols] = result[cat_cols].astype("category")

print("Setting numeric cols...")
num_cols = configcols.NUMERIC_COLS
result[num_cols] = result[num_cols].apply(pd.to_numeric, errors="coerce")
result[num_cols] = result[num_cols].round(2)

print("Dropping empty columns...")
print(len(list(result)))
# drop columns if > 99.9% null
print(
    list(result.loc[:, result.isnull().sum() > 0.999 * result.shape[0]])
)  # print the cols that are >99.9% null
result = result.loc[
    :, result.isnull().sum() < 0.999 * result.shape[0]
]  # keep the ones that are <99.9% null
print(len(list(result)))

# Save features to csv
import time

timestrfolder = time.strftime("%Y-%m-%d")
datafolder = config.PROCESSED_DATA_DIR / timestrfolder
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
feature_list = list(result)
df = pd.DataFrame(feature_list, columns=["features"])
spreadsheet_title = "Feature list 03 clean09 "
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".csv"
title = spreadsheet_title + timestr + ext
feature_list_file = datafolder / title
df.to_csv(feature_list_file, index=False)
print("CSV of features available at: ", feature_list_file)

# Save pickle
result_file = config.CLEAN_PHASE_09
result.to_pickle(result_file)
print("File available at :", result_file)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
