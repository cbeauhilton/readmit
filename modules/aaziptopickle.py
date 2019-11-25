from datetime import datetime

startTime = datetime.now()

import os
import sys
import zipfile
from io import StringIO
from tqdm import tqdm

tqdm.pandas()

import pandas as pd

sys.path.append("modules")
from cbh import config

archive = config.RAW_ZIP_FILE

print("About to run ", os.path.basename(__file__))

print("Extracting zip file...")
zip_ref = zipfile.ZipFile(archive, "r")
extracted = zip_ref.namelist()
zip_ref.extractall(config.RAW_DATA_DIR)
zip_ref.close()
extracted_file = os.path.join(config.RAW_DATA_DIR, extracted[0])
print("Zip file extracted.")

print("Reading text file into dataframe...")
ccf_raw = pd.read_csv(extracted_file, sep="\t", header=0, lineterminator='\r')
print("Dataframe created.")

print("Basic text cleaning for column names...")
ccf_raw.columns = (
    ccf_raw.columns.str.strip()
    .str.lower()
    .str.replace("  ", "_")
    .str.replace(" ", "_")
    .str.replace("__", "_")
)

# print("Fixing indices...")
# The "patientID" column is the index

ccf_raw.dropna(subset=['encounterid'], inplace=True)
ccf_raw.dropna(subset=['patientid'], inplace=True)
ccf_raw['encounterid'] = ccf_raw['encounterid'].apply(str)
# print(list(ccf_raw))
print(ccf_raw)

print("Setting datetime columns to correct dtypes...")

ccf_raw = ccf_raw.progress_apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)


# Using "patientid" as index results in duplicates
# this will sort the dataframe by admission time
# and assign an index value starting from the first admission
# and pull out the "patientid" column
ccf_raw = ccf_raw.sort_values(["admissiontime"])


# All the pts admitted in 2010 have super long stays -
# I'm guessing the pull was for _discharges_ starting in 2011
# so any admissions starting in 2010 would be old to super old
# Data degrades after May 2018 (not as many admissions, LoS drops surreptitiously)
print(len(ccf_raw))
ccf_raw = ccf_raw[
    (ccf_raw["admissiontime"] > "2011-01-01")
    # & (ccf_raw["admissiontime"] < "2019-01-01") 
    & (ccf_raw["dischargetime"] < "2018-05-01")
]
print(len(ccf_raw))

print("Generating length of stay...")
# the "length_of_stay" column provided by CCF does not provide for partial days
# so calculate it fresh

ccf_raw["admissiontime"] = pd.to_datetime(ccf_raw["admissiontime"])
ccf_raw["length_of_stay_in_time"] = ccf_raw["dischargetime"] - ccf_raw["admissiontime"]
ccf_raw["length_of_stay_in_days"] = ccf_raw.progress_apply(
    lambda row: row.length_of_stay_in_time.total_seconds(), axis=1
)
# divide seconds by 3600 to get hours
ccf_raw["length_of_stay_in_days"] = ccf_raw["length_of_stay_in_days"] / 3600
# divide hours by 24 to get days
# doing it this way (as opposed to .days method) gives partial days
# the two divisions is just to make it explicit
ccf_raw["length_of_stay_in_days"] = ccf_raw["length_of_stay_in_days"] / 24

# drop all rows with missing length of stay
ccf_raw = ccf_raw[pd.notnull(ccf_raw["length_of_stay_in_days"])]

# generate boolean length of stay thresholds for binary analyses
print("Generating LoS binary columns...")

ccf_raw["length_of_stay_over_3_days"] = ccf_raw[
    "length_of_stay_in_days"
].progress_apply(lambda x: 1 if x > 3 else 0)

ccf_raw["length_of_stay_over_5_days"] = ccf_raw[
    "length_of_stay_in_days"
].progress_apply(lambda x: 1 if x > 5 else 0)

ccf_raw["length_of_stay_over_7_days"] = ccf_raw[
    "length_of_stay_in_days"
].progress_apply(lambda x: 1 if x > 7 else 0)

ccf_raw["length_of_stay_over_14_days"] = ccf_raw[
    "length_of_stay_in_days"
].progress_apply(lambda x: 1 if x > 14 else 0)


print("Generating time since last discharge...")
new_column = ccf_raw.groupby("patientid", as_index=False).progress_apply(
    lambda x: x["admissiontime"] - x["dischargetime"].shift(1)
)
ccf_raw[
    "time_between_current_admission_and_previous_discharge"
] = new_column.reset_index(
    level=0, drop=True
) 
# I'm not sure about the "reset_index" thing here, but pd was throwing errors without it and
# checking the numbers for a bunch of patients shows it works fine.

print("Generating time to next admission...")
# This will yield a negative time delta, fixed below
# when "days_between_current_discharge_and_next_admission" is generated.
new_column1 = ccf_raw.groupby("patientid", as_index=False).progress_apply(
    lambda x: x["dischargetime"] - x["admissiontime"].shift(-1)
)
ccf_raw["time_between_current_discharge_and_next_admission"] = new_column1.reset_index(
    level=0, drop=True
)

print("Probably another useless pd.to_datetime command...")
ccf_raw = ccf_raw.progress_apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)


# This will let me do "length of stay of last admission" easily
print(
    "Generating time between beginning of current admission and _beginning_ of last admission (for _last LoS_ calculation)..."
)

ccf_raw["time_since_beginning_of_last_admission"] = (
    ccf_raw.sort_values(["patientid", "admissiontime"])
    .groupby("patientid")["admissiontime"]
    .diff()
)

# these next ones convert the timedeltas
# to integers so they will play nicely in the models
print("Days since beginning of last admission...")
ccf_raw["days_since_beginning_of_last_admission"] = ccf_raw.progress_apply(
    lambda row: row.time_since_beginning_of_last_admission.total_seconds(), axis=1
)

ccf_raw["days_since_beginning_of_last_admission"] = (
    ccf_raw["days_since_beginning_of_last_admission"] / 3600
)
ccf_raw["days_since_beginning_of_last_admission"] = (
    ccf_raw["days_since_beginning_of_last_admission"] / 24
)

print("Days since last discharge...")
ccf_raw[
    "days_between_current_admission_and_previous_discharge"
] = ccf_raw.progress_apply(
    lambda row: row.time_between_current_admission_and_previous_discharge.total_seconds(),
    axis=1,
)
ccf_raw["days_between_current_admission_and_previous_discharge"] = (
    ccf_raw["days_between_current_admission_and_previous_discharge"] / 3600
)
ccf_raw["days_between_current_admission_and_previous_discharge"] = (
    ccf_raw["days_between_current_admission_and_previous_discharge"] / 24
)

ccf_raw["length_of_stay_of_last_admission"] = (
    ccf_raw["days_between_current_admission_and_previous_discharge"]
    - ccf_raw["days_since_beginning_of_last_admission"]
)

print("Days until next admission...")
ccf_raw["days_between_current_discharge_and_next_admission"] = ccf_raw.progress_apply(
    lambda row: row.time_between_current_discharge_and_next_admission.total_seconds(),
    axis=1,
)
# Multiply by negative 1 to get a positive number
ccf_raw["days_between_current_discharge_and_next_admission"] = (
    ccf_raw["days_between_current_discharge_and_next_admission"] * -1
)
ccf_raw["days_between_current_discharge_and_next_admission"] = (
    ccf_raw["days_between_current_discharge_and_next_admission"] / 3600
)
ccf_raw["days_between_current_discharge_and_next_admission"] = (
    ccf_raw["days_between_current_discharge_and_next_admission"] / 24
)
print("...done.")

print("Saving to pickle...")
file_name = config.RAW_DATA_FILE
ccf_raw.to_pickle(file_name)
print("Pickle file available at", file_name)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")


# # fix values wrt casing, bad spacing, etc.
# print("Cleaning text within cells...")
# ccf_raw = ccf_raw.progress_apply(
#     lambda x: x.str.lower()
#     .str.strip()
#     .str.replace("\t", "")
#     .str.replace("  ", " ")
#     .str.replace(" ", "_")
#     .str.replace("__", "_")
#     if (x.dtype == "object")
#     else x
# )
