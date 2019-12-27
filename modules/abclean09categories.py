import json
import os
import sys
import zipfile
from datetime import datetime
from io import StringIO
from urllib.request import urlopen

from pathlib import Path
import numpy as np

import pandas as pd
from pandas.io.json import json_normalize
from tqdm import tqdm

tqdm.pandas()

sys.path.append("modules")
from cbh import config
from cbh import configcols

print("About to run", os.path.basename(__file__))

startTime = datetime.now()

# data_file = config.CLEAN_PHASE_04
# print("Loading", data_file)
# data = pd.read_pickle(data_file)

interim_file = config.INTERIM_H5
print("Loading", interim_file)
data = pd.read_hdf(interim_file, key='phase_04')
print("File loaded.")

print(list(data))

before_dedup = len(data)
# print(data["encounterid"])
data = data.drop_duplicates(subset=["encounterid"], keep="first")
after_dedup = len(data)
print(f"Dropped {before_dedup - after_dedup} duplicate encounters")

data = data[data.patient_age < 130]  # https://en.wikipedia.org/wiki/Oldest_people
data = data[data.patient_age >= 0]
agism = len(data)
# negative ages snuck in somehow. Using (>= 0) allows newborns.
print(f"Dropped {after_dedup - agism} impossibly aged encounters")

# Define readmission at various thresholds
readmission_thresholds = [3, 5, 7, 15, 20, 28, 30, 45, 90, 180, 365, 3650]
for thresh in readmission_thresholds:
    print(f"Making column for readmission threshold at {thresh} days...")
    data[f"readmitted{thresh}d"] = (
        data["days_between_current_discharge_and_next_admission"] <= thresh
    ) & (
        data["days_between_current_discharge_and_next_admission"] > 0.15
    )  # adding 4 hours accounts for transfers, based on histogram analysis
    data[f"readmitted{thresh}d"] = (
        data[f"readmitted{thresh}d"] * 1
    )  # convert to 1/0 rather than True/False

print(data["acs_200_and_over_ratio_income_poverty_level_past_12_mo"])

# Mobilize some of the ACS data
data["over_200_ratio"] = (
    data["acs_200_and_over_ratio_income_poverty_level_past_12_mo"]
    / data["acs_total_population_count"]
)
data["white_alone_ratio"] = (
    data["acs_race_white_alone"] / data["acs_total_population_count"]
)

# BMI delta
data["BMI_delta"] = pd.to_numeric(
    data["bmi_discharge"], errors="coerce"
) - pd.to_numeric(data["bmi_admit"], errors="coerce")
# print(data["BMI_delta"])

# BP deltas
data["discharge_diastolic_delta"] = pd.to_numeric(
    data["discharge_diastolic_bp"], errors="coerce"
) - pd.to_numeric(data["admit_diastolic_bp"], errors="coerce")
data["discharge_systolic_delta"] = pd.to_numeric(
    data["discharge_systolic_bp"], errors="coerce"
) - pd.to_numeric(data["admit_systolic_bp"], errors="coerce")
# print(data.discharge_diastolic_delta)

print("Binarizing patient age...")
age_thresholds = [10, 30, 65]
for thresh in age_thresholds:
    print(f"Making binary column for age at {thresh} years...")
    data[f"age_gt_{thresh}_y"] = data["patient_age"] >= thresh
    # convert to 1/0 rather than True/False
    data[f"age_gt_{thresh}_y"] = data[f"age_gt_{thresh}_y"] * 1
    

print("Converting diagnosis codes to diagnosis descriptions as available...")
dxcode_file = config.DX_CODES_CONVERTED
dxcodes = pd.read_pickle(dxcode_file)
di = pd.Series(
    dxcodes["diagnosis description"].values, index=dxcodes["diagnosis code"]
).to_dict()

# data["primary_diagnosis_code"].replace(di, inplace=True) # super slow - use map instead
# https://stackoverflow.com/questions/20250771/remap-values-in-pandas-column-with-a-dict

data["primary_diagnosis_code"] = (
    data["primary_diagnosis_code"].map(di).fillna(data["primary_diagnosis_code"])
)

print("Dropping negative LoS rows (n=97, all same-day)...")
data = data[data["length_of_stay_in_days"] >= 0]

print(
    "Dropping encounters with a discharge disposition of Expired and somehow were also readmitted within 30d (n=43)"
)
print("Before", len(data))
data = data.drop(
    data[
        (
            (data.dischargedispositiondescription == "Expired")
            & (data.readmitted30d > 0)
        )
    ].index
)
print("After", len(data))

datecols = data.filter(items=configcols.DATETIME_COLS)

print("Dropping datetime cols...")
data = data.drop(configcols.DATETIME_COLS, axis=1, errors="ignore")

print("Dropping empties...")
# 1 of 2 - drop before making categoricals
data = data.loc[
    :, data.isnull().sum() < 0.999 * data.shape[0]
]  # keep the ones that are <99.9% null


# fix values wrt casing, bad spacing, etc.
print("Cleaning text within cells...")
data = data.progress_apply(
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

print("Setting categorical cols astype _category_...")
cat_cols = configcols.CATEGORICAL_COLS
for col in cat_cols:
    try:
        data[col] = data[col].astype("category")
    except:
        print(f"{col} not in cols. But that's OK.")

print("Setting numeric cols...")
num_cols = configcols.NUMERIC_COLS
for col in num_cols:
    try:
        data[col] = data[col].apply(pd.to_numeric, errors="coerce", downcast="signed")
        data[col] = data[col].round(2)
    except:
        print(f"{col} not in cols. But that's OK.")

nums = data.select_dtypes(include=np.unsignedinteger).columns.tolist()
for col in nums:
    try:
        print(col)
        data[col] = data[col].apply(pd.to_numeric, downcast="signed")
    except:
        print(f"Could not convert {col} to signed.")

print("Merging...")
data = pd.merge(data, datecols, left_index=True, right_index=True)


print("Dropping empty columns...")
# 2 of 2 - drop any cols rendered empty by previous operations
print(len(list(data)))
# drop columns if > 99.9% null
print(
    list(data.loc[:, data.isnull().sum() > 0.999 * data.shape[0]])
)  # print the cols that are >99.9% null
data = data.loc[
    :, data.isnull().sum() < 0.999 * data.shape[0]
]  # keep the ones that are <99.9% null
print(len(list(data)))

# Save features to csv
import time

timestrfolder = time.strftime("%Y-%m-%d")
datafolder = config.PROCESSED_DATA_DIR / timestrfolder
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
feature_list = list(data)
df = pd.DataFrame(feature_list, columns=["features"])
spreadsheet_title = "Feature list 03 clean09 "
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".csv"
title = spreadsheet_title + timestr + ext
feature_list_file = datafolder / title
df.to_csv(feature_list_file, index=False)
print("CSV of features available at: ", feature_list_file)

# print(data.dtypes.to_dict())
# with open(config.PROCESSED_DATA_DIR/"dtypes.json", 'w') as json_file:
#         json.dump(data.dtypes.to_dict(), json_file)
#         data.dtypes.to_json()
# print(data.select_dtypes(include=['uint64']))

# Save to disk
# '09' just in case I want to go add some other preprocessing later
# o BASIC, thy wisdom shines eternal.

result_file = config.CLEAN_PHASE_09
data.index = data.index.astype('float64') 
data.to_pickle(result_file)
try:
    data.to_hdf(interim_file, key='phase_09', mode='a', format='table')
except:
    print("HDF table format didn't work, trying fixed...")
else:
    data.to_hdf(interim_file, key='phase_09', mode='a', format='fixed')
print(f"Files available at : {result_file} and {interim_file}")

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
