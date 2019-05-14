# Imports
import sys

sys.path.append("modules")

import config
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from tqdm import tqdm

tqdm.pandas()

from datetime import datetime

startTime = datetime.now()

pd.options.display.max_columns = 2000

# fix random seed for reproducibility
seed = config.SEED
np.random.seed(seed)

# Load file
filename = config.RAW_DATA_FILE

print("Loading", filename)
data = pd.read_pickle(filename)

print("File loaded.")

print("Fixing indices...")
# The "patientID" column is the index
data = data.rename_axis("patientid")

# Using "patientid" as index results in duplicates
# this will sort the dataframe by admission time
# and assign an index value starting from the first admission
# and pull out the "patientid" column
data = data.sort_values(["admissiontime"]).reset_index()

# FOR DEBUGGING
# comment out the next section to run the real thing
# print("Selecting small portion for debugging...")
# data = data[:20000]

# fix values wrt casing, bad spacing, etc.
print("Cleaning text within cells...")
data = data.progress_apply(
    lambda x: x.str.lower()
    .str.strip()
    .str.replace("\t", "")
    .str.replace("  ", " ")
    .str.replace(" ", "_")
    .str.replace("__", "_")
    if (x.dtype == "object")
    else x
)

print("Filling NA with 0 as appropriate...")
data.update(
    data[
        [
            "ed_admission",
            "diff_type_discharge_value",
            "lineinfection",
            "cdiffinfection",
            "fallduringadmission",
            "ondialysis",
            "opiatesduringadmit",
            "benzosduringadmit",
            "pressureulcer",  # not binary - fix below to avoid mixed dtypes
            "dischargedonbenzo",
            "dischargedonopiate",
            "myocardial_infarction",
            "congestive_heart_failure",
            "peripheral_vascular_disease",
            "cerebrovascular_disease",
            "dementia",
            "chronic_pulmonary_disease",
            "rheumatic_disease",
            "peptic_ulcer_disease",
            "mild_liver_disease",
            "diabetes_without_chronic_complication",
            "diabetes_with_chronic_complication",
            "hemiplegia_or_paraplegia",
            "renal_disease",
            "any_malignancy,including_lymphoma_and_leukemia,except_malignant_neoplasm_of_skin",
            "moderate_or_severe_liver_disease",
            "metastatic_solid_tumor",
            "aids/hiv",
            "connective_tissue_disorder",
            "pneumonia",
            "depression",
            "anxiety",
            "psychosis",
            "cerebral_palsy",
            "short_gut_syndrome",
            "epilepsy",
            "knee_replacement",
            "hip_replacement",
            "solid_organ_transplant",
            "tpn",
            "pt_ot_consult",
            "spiritualcareconsult",
            "palliativecareconsult",
            "infectiousdiseaseconsult",
            "hospiceconsult",
        ]
    ].fillna(0)
)

data["pressureulcer"] = data["pressureulcer"].replace([0, "n"], "no")

# split bp into two columns(sys/dia), first for admit then discharge
new = data["admitbp"].str.split("/", n=1, expand=True)

# making systolic column from new data frame
data["admit_systolic_bp"] = new[0]

# making diastolic column from new data frame
data["admit_diastolic_bp"] = new[1]

# Dropping old bp column
data.drop(columns=["admitbp"], inplace=True)

# and again for discharge
new = data["dischargebp"].str.split("/", n=1, expand=True)
data["discharge_systolic_bp"] = new[0]
data["discharge_diastolic_bp"] = new[1]
data.drop(columns=["dischargebp"], inplace=True)

# fill NaNs more thoroughly
data.replace(to_replace=[None], value=np.nan, inplace=True)
data.fillna(value=pd.np.nan, inplace=True)

# Sort by admissions per patient
data = data.sort_values(["patientid", "admissiontime"]).reset_index()

# Count number of past admissions for each admission for each patient
data["number_past_admissions"] = data.groupby("patientid").cumcount()

print("Fixing dtypes...")
data_bool = data.select_dtypes(["bool"])
converted_bool = data_bool * 1.0  # changes bool to int

data_int = data.select_dtypes(include=["int"])
converted_int = data_int.progress_apply(pd.to_numeric, downcast="unsigned")

data_float = data.select_dtypes(include=["float"])
converted_float = data_float.progress_apply(pd.to_numeric, downcast="float")

data[converted_int.columns] = converted_int
data[converted_float.columns] = converted_float
data[converted_bool.columns] = converted_bool

data_obj = data.select_dtypes(
    include=["object"], exclude=["datetime", "timedelta"]
).copy()
converted_obj = pd.DataFrame()
for col in data_obj.columns:
    num_unique_values = len(data_obj[col].unique())
    num_total_values = len(data_obj[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:, col] = data_obj[col].astype("category")
    else:
        converted_obj.loc[:, col] = data_obj[col]

data[converted_obj.columns] = converted_obj


print("Generating readmission timing - this might take a while...")

print("Generating time since last discharge...")
new_column = data.groupby("patientid", as_index=False).progress_apply(
    lambda x: x["admissiontime"] - x["dischargetime"].shift(1)
)
data["time_between_current_admission_and_previous_discharge"] = new_column.reset_index(
    level=0, drop=True
)

print("Generating time to next admission...")
# This will yield a negative time delta, fixed below when #days is generated.
new_column1 = data.groupby("patientid", as_index=False).progress_apply(
    lambda x: x["dischargetime"] - x["admissiontime"].shift(-1)
)
data["time_between_current_discharge_and_next_admission"] = new_column1.reset_index(
    level=0, drop=True
)

# make possibly interesting time variables
print("Generating other time variables...")

data["admit_date"] = data["admissiontime"].dt.date
data = data.progress_apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)
data["admit_day_of_week"] = data["admissiontime"].dt.weekday_name
data["discharge_day_of_week"] = data["dischargetime"].dt.weekday_name
data["admission_hour_of_day"] = data["admissiontime"].dt.hour
data["discharge_hour_of_day"] = data["dischargetime"].dt.hour
data["admit_year"] = data["admissiontime"].dt.year
data["discharge_year"] = data["dischargetime"].dt.year
cal = calendar()
holidays = cal.holidays(start="2000-01-01", end="2050-12-31")
data["admitted_on_holiday"] = data["admissiontime"].isin(holidays)
data["discharged_on_holiday"] = data["dischargetime"].isin(holidays)
data["length_of_stay"] = data["dischargetime"] - data["admissiontime"]
print("Generating length of stay...")
data["length_of_stay_in_days"] = data.progress_apply(
    lambda row: row.length_of_stay.days, axis=1
)


data["time_since_beginning_of_last_admission"] = (
    data.sort_values(["patientid", "admissiontime"])
    .groupby("patientid")["admissiontime"]
    .diff()
)


# these next ones convert timedeltas
# to integers to play nicely in the models
print("Days since beginning of last admission...")
data["days_since_beginning_of_last_admission"] = data.progress_apply(
    lambda row: row.time_since_beginning_of_last_admission.days, axis=1
)

print("Days since last discharge...")
data["days_between_current_admission_and_previous_discharge"] = data.progress_apply(
    lambda row: row.time_between_current_admission_and_previous_discharge.days, axis=1
)

print("Days until next admission...")
data["days_between_current_discharge_and_next_admission"] = data.progress_apply(
    lambda row: row.time_between_current_discharge_and_next_admission.days, axis=1
)
# Multiply by negative 1 to get a positive number
data["days_between_current_discharge_and_next_admission"] = (
    data["days_between_current_discharge_and_next_admission"] * -1
)


# Not sure if these next two are necessary anymore
data = data.progress_apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)

data = data.progress_apply(
    lambda col: pd.to_timedelta(col, errors="ignore")
    if (col.dtypes == object)
    else col,
    axis=0,
)

# this one got categorized - check if
# change to data_obj worked, this line might not be needed
# data["length_of_stay"] = data["dischargetime"] - data["admissiontime"]

# make a new index by admission time
data = data.sort_values(["admissiontime"]).reset_index(drop=True)

# and wrap it all up in a file, pickle or h5
# print("Saving to file...")
# filename1 = config.CLEAN_PHASE_00
# key = config.CLEAN_PHASE_00_KEY
# data.to_hdf(filename1, key=key, format="table")
# print("Clean phase_00 available at:", filename1, "via key", key)

print("Saving to file...")
filename1 = config.CLEAN_PHASE_00
data.to_pickle(filename1)
print("Clean phase_00 available at:", filename1)

# FOR DEBUGGING
# move this section around wherever you want to save interim results
file_name = config.INTERIM_DATA_DIR / "data_DEBUG.pickle"
data.to_pickle(file_name)
print("Debug data file saved.")

# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
