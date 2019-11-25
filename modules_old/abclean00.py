# Imports
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from fastparquet import write
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from tqdm import tqdm

import configcols
from cbh import config

tqdm.pandas()

print("About to run", os.path.basename(__file__))
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

# FOR DEBUGGING
# comment out the next section to run the real thing
# print("Selecting small portion for debugging...")
# data = data[:20000]

# Fix some ugly column names
data = data.rename(
    index=str,
    columns={
        'any_malignancy,_including_lymphoma_and_leukemia,_except_malignant_neoplasm_of_skin': "any_cancer_not_skin",  # commas and verbosity
        "aids/hiv": "aids_hiv",  # yikes, "/" make things difficult
        # Rename the ACS data so it's interpretable:
        "B01003_001E": "acs_total_population_count",
        "B02001_002E": "acs_race_white_alone",
        "B02001_003E": "acs_race_black_alone",
        "B11001_003E": "acs_married_couple_family",
        "C17002_002E": "acs_under_50_ratio_income_poverty_level_past_12_mo",
        "C17002_003E": "acs_50_to_99_ratio_income_poverty_level_past_12_mo",
        "C17002_004E": "acs_100_to_124_ratio_income_poverty_level_past_12_mo",
        "C17002_005E": "acs_125_to_149_ratio_income_poverty_level_past_12_mo",
        "C17002_006E": "acs_150_to_184_ratio_income_poverty_level_past_12_mo",
        "C17002_007E": "acs_185_to_199_ratio_income_poverty_level_past_12_mo",
        "C17002_008E": "acs_200_and_over_ratio_income_poverty_level_past_12_mo",
    },
)

# print(list(data))
print("Filling NA with 0 as appropriate...")
data.update(data[configcols.FILL_NA_WITH_ZERO_COLS].fillna(0))

print("Making data prettier and more compact...")
data["pressureulcer"] = data["pressureulcer"].replace([0, "n"], "No")
# There were only a handful of "surgical incision"
# and "hospital acquired" ulcers coded as such
data["pressureulcer"].replace(
    {"Surgical Incision": "Yes", "Hospital Acquired": "Yes"}, inplace=True
)

print("For race...")
data["race"].replace(
    {
        "American Indian or Alaska Native": "Other",
        "Asian or Pacific islander": "Other",
        "Asians": "Other",
        "Hispanic Americans": "Other",
        "Multiracial": "Other",
        "Native Hawaiian or Other Pacific Islander": "Other",
        "Race: [other] or [not stated]": "Other",
    },
    inplace=True,
)
# Copy col
data["race_binary"] = data.race
# Rename values
data["race_binary"].replace({"Black": 1, "Other": 1, "White": 0}, inplace=True)
# Convert to numeric
data.race_binary = pd.to_numeric(data.race_binary, errors="coerce")

data["race"] = data["race"].astype("category")

print("For financialclass...")
data["financialclass"].replace(
    {
        "Blue Cross": "Other",
        "Blue Cross Blue Shield Insurance Plans": "Other",
        "Blue Shield": "Other",
        "Government Dignitary": "Other",
        "Managed Care": "Other",
        "Employee Health Insurance": "Other",
        "Military Personnel": "Other",
        "SELFPAY": "Other",
        "Worker Compensation": "Other",
        "International Agencies": "Other",
    },
    inplace=True,
)

data["financialclass_binary"] = data["financialclass"]
data["financialclass_binary"].replace(
    {"Other": 0, "Medicaid": 1, "Medicare": 1, "Private Health Insurance": 0},
    inplace=True,
)
data.financialclass_binary = pd.to_numeric(data.financialclass_binary, errors="coerce")


print("For patientclassdescription...")
data["patientclassdescription"].replace(
    {
        "Hospital admission,short-term,day care": "Other",
        "Psychiatry Specialty": "Other",
        "Rehabilitation - specialty": "Other",
        "Specimen": "Other",
        "Therapeutic procedure": "Other",
        "Newborn,nursery,infants": "Other",
        "patient scheduled for surgery": "Other",
        "long-term care": "Other",
        "patient scheduled for surgery": "Other",
        "Patient Class - Emergency": "Emergency",
        "Patient Class - Outpatient": "Outpatient",
        "hospice patient": "Hospice",
    },
    inplace=True,
)
data["patientclassdescription"] = data["patientclassdescription"].astype("category")

print("For dischargedispositiondescription...")
data["dischargedispositiondescription"].replace(
    {
        "Hospice care provided in inpatient hospice facility": "Hospice",
        "Hospice Care (CMS Temporary Codes)": "Hospice",
        "Discharged/transferred to another short term general hospital for inpatient care": "Transfer to another hospital",
        "Hospitals,Federal": "Transfer to another hospital",
        "Hospitals,Pediatric": "Transfer to another hospital",
        "Patient transfer,to another health care facility ": "Transfer to another hospital",
        "Discharge to police custody": "Other",
        "Active Inpatient": "Other",
        "patient scheduled for surgery": "Other",
        "intravenous administration": "Other",
        "Other error requiring error requiring inactivation": "Other",
        "Critical Access - NUCCProviderCodes": "Other",
        "Patient Readmission": "Other",
        "Patient transfer,to another health care facility": "Other",
        "Psychiatric hospital": "Transfer to a psychiatric hospital",
        "skilled nursing facility": "Skilled nursing facility",
        # 'Long-Term Care Facility' : "SNF or LTC",
        "Nursing & Custodial Care Facilities; Nursing Facility/Intermediate Care Facility": "Intermediate care facility",
        "Death (finding)": "Expired",
        "Person location type - Home": "Home",
        "General Acute Care Hospital - NUCCProviderCodes": "General Acute Care Hospital",
    },
    inplace=True,
)
data["dischargedispositiondescription"] = data[
    "dischargedispositiondescription"
].astype("category")

print("For marital_status...")
data["marital_status"].replace(
    {
        "Currently Married": "Married or partnered",
        "domestic partner": "Married or partnered",
        "Unmarried person": "Single",
        "Separated from cohabitee": "Divorced or separated",
        "Divorced state": "Divorced or separated",
        "Marital state unknown": "Other",
        "Other - Marital Status": "Other",
        "Patient data refused": "Other",
        "Widow": "Widowed",
    },
    inplace=True,
)
data["marital_status"] = data["marital_status"].astype("category")


data["gender_binary"] = data["gender"]
# male has to be 1 to get the proportions right
data["gender_binary"].replace({"Male": 1, "Female": 0}, inplace=True)
data.gender_binary = pd.to_numeric(data.gender_binary, errors="coerce")
data["gender"] = data["gender"].astype("category")


print("Getting dummies for pressureulcer and financialclass...")
data = pd.get_dummies(data=data, dummy_na=True, prefix=None, columns=["pressureulcer"])
# Copy financial class first so we can use it in TableOne
data["financialclass_orig"] = data.financialclass
data = pd.get_dummies(data=data, dummy_na=True, prefix=None, columns=["financialclass"])
# data = pd.get_dummies(data=data, dummy_na=True, prefix=None, columns=["patientclassdescription"])

print(
    "Filling nones with NaNs..."
)  # make sure to do this AFTER filling with zeros as appropriate (above)
# fill missing values with NaNs
data.replace(to_replace=[None], value=np.nan, inplace=True)
data.fillna(value=pd.np.nan, inplace=True)

print("Fixing BP columns...")
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


# Sort by admissions per patient
data = data.sort_values(["patientid", "admissiontime"])  # probably redundant

print("Counting number of past admissions for each admission for each patient...")
data["number_past_admissions"] = data.groupby("patientid").cumcount()

print("Getting number of morbidities per patient...")
morbidity_binary_cols = configcols.MORBIDITY_BINARY_COLS
data["number_morbidities"] = data[morbidity_binary_cols].sum(axis=1)

# make possibly interesting time variables
print("Generating other time variables...")

# Make a dummy column
# with "1" for each patient encounter
# for doing burden and frequency analysis, making pretty pictures
data["this_is_hospital_encounter"] = 1

# These are actually "times," not "dates."
# These name changes will make the results of datetime feature engineering below prettier.
data = data.rename(
    index=str,
    columns={
        "cbc_admit_date": "cbc_admit_time",
        "cbc_discharge_date": "cbc_discharge_time",
        "cmp_admit_date": "cmp_admit_time",
        "cmp_discharge_date": "cmp_discharge_time",
    },
)

timecols = [
    "admissiontime",
    "dischargetime",
    "cbc_admit_time",
    "cbc_discharge_time",
    "cmp_admit_time",
    "cmp_discharge_time",
]

for timecol in timecols:
    cal = calendar()  # initialize US federal holiday calendar
    holidays = cal.holidays(start="2000-01-01", end="2050-12-31")
    print(f"Engineering features for {timecol}...")
    data[f"{timecol}_date"] = data[timecol].dt.date
    data[f"{timecol}_on_holiday"] = data[f"{timecol}_date"].isin(holidays)
    data[f"{timecol}_day_of_week"] = data[timecol].dt.weekday_name
    data[f"{timecol}_weekofyear"] = data[timecol].dt.weekofyear
    data[f"{timecol}_month"] = data[timecol].dt.month
    data[f"{timecol}_quarter"] = data[timecol].dt.quarter
    data[f"{timecol}_year"] = data[timecol].dt.year
    data[f"{timecol}_hour_of_day"] = data[timecol].dt.hour
    data[f"{timecol}_during_working_day_7a_6p"] = (data[timecol].dt.hour > 7) & (
        data[timecol].dt.hour < 18
    )
    data[f"{timecol}_at_night_6p_10p"] = (data[timecol].dt.hour < 22) & (
        data[timecol].dt.hour > 18
    )
    data[f"{timecol}_late_night_10p_12a"] = (data[timecol].dt.hour < 24) & (
        data[timecol].dt.hour > 22
    )
    data[f"{timecol}_very_late_night_12a_4a"] = (data[timecol].dt.hour < 4) & (
        data[timecol].dt.hour > 0
    )
    data[f"{timecol}_early_morning_4a_7a"] = (data[timecol].dt.hour < 7) & (
        data[timecol].dt.hour > 4
    )
    # number of pts discharged/admitted on the same day as the pt (idea of total hospital burden)
    # see https://stackoverflow.com/questions/30244952/python-pandas-create-new-column-with-groupby-sum
    if timecol == "admissiontime" or "dischargetime":
        data[f"{timecol}_hospital_burden"] = (
            data["this_is_hospital_encounter"]
            .groupby(data[timecol].dt.date)
            .transform("sum")
        )

    if (
        timecol == "cbc_admit_time"
        or "cbc_discharge_time"
        or "cmp_admit_time"
        or "cmp_discharge_time"
    ):
        # print(data[timecol])
        # 2018-04-23 17:26:00 "%Y-%m-%d %H:%M:%S"
        # data[timecol] = pd.to_datetime(timecol, format="%Y-%m-%d %H:%M:%S")
        data[f"{timecol}_within_24h"] = (
            data[timecol].dt.date - data["admissiontime"].dt.date
        ).dt.total_seconds()  # .astype('timedelta64[s]')
        data[f"{timecol}_within_24h"] = data[f"{timecol}_within_24h"] < (60 * 60 * 24)*1
        print(f"{timecol}_within_24h")
        # print(data[f"{timecol}_within_24h"])
    else:
        pass

    weekend = {"Saturday": 1, "Sunday": 1}
    weekday = {"Monday": 0, "Tuesday": 0, "Wednesday": 0, "Thursday": 0, "Friday": 0}

    data[f"{timecol}_on_weekend"] = data[f"{timecol}_day_of_week"]
    data[f"{timecol}_on_weekend"] = data[f"{timecol}_on_weekend"].replace(weekend)
    data[f"{timecol}_on_weekend"] = data[f"{timecol}_on_weekend"].replace(weekday)

    # for easy copy-pasta
    print(f"{timecol}_at_night_6p_10p")
    print(f"{timecol}_date")
    print(f"{timecol}_day_of_week")
    print(f"{timecol}_during_working_day_7a_6p")
    print(f"{timecol}_early_morning_4a_7a")
    print(f"{timecol}_hospital_burden")
    print(f"{timecol}_hour_of_day")
    print(f"{timecol}_late_night_10p_12a")
    print(f"{timecol}_month")
    print(f"{timecol}_on_holiday")
    print(f"{timecol}_quarter")
    print(f"{timecol}_very_late_night_12a_4a")
    print(f"{timecol}_weekofyear")
    print(f"{timecol}_year")

# Define readmission at various thresholds
readmission_thresholds = [3, 5, 7, 15, 20, 28, 30, 45, 90, 180, 365, 3650]
for thresh in readmission_thresholds:
    print(f"Making column for readmission threshold at {thresh} days...")
    data[f"readmitted{thresh}d"] = (
        data["days_between_current_discharge_and_next_admission"] <= thresh
    ) & (
        data["days_between_current_discharge_and_next_admission"] > 0.1
    )  # adding one accounts for transfers
    data[f"readmitted{thresh}d"] = (
        data[f"readmitted{thresh}d"] * 1
    )  # convert to 1/0 rather than True/False

# Readmission thresholds in Artetxe et al 2018:
# <3d AUC 0.66-0.79
# 07d AUC 0.74-0.82
# 15d AUC 0.65-0.70
# 28d AUC 0.60-0.92
# 30d AUC 0.60-0.92
# 45d AUC 0.69-0.69
# 90d AUC 0.65-0.65
# 180 AUC 0.65-0.76
# 365 AUC 0.70-0.77

# Target populations:
# All-cause
# HF
# Age < 65
# AMI
# Pneumonia

# Find patients who were probably transferred
data["probably_a_transfer"] = (
    data["days_between_current_admission_and_previous_discharge"] < 0.5
) * 1  # convert to 1/0 rather than True/False

# Change the "days_between..." column to only include probable non-transfers,
# else fill with "NaN"
a = np.array(
    data["days_between_current_admission_and_previous_discharge"].values.tolist()
)
data["days_between_current_admission_and_previous_discharge"] = np.where(
    a < 0.5, np.nan, a
).tolist()

data["discharged_in_past_30d"] = (
    data["days_between_current_admission_and_previous_discharge"] < 30
) * 1  # convert to 1/0 rather than True/False



# data['dateofbirth'] = pd.to_datetime(data["dateofbirth"])
data = data.progress_apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)


print("Fixing indices...")
# The "patientID" column is the index
data = data.rename_axis("patientid_")

# Using "patientid" as index results in duplicates
# this will sort the dataframe by admission time
# and assign an index value starting from the first admission
# and pull out the "patientid" column
data = data.sort_values(["admissiontime"]).reset_index()

# FOR DEBUGGING
# comment out the next section to run the real thing
# print("Selecting small portion for debugging...")
# data = data[:20000]


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


print("Fixing dtypes...")
data_bool = data.select_dtypes(["bool"])
converted_bool = data_bool * 1.0  # changes bool to int

data_int = data.select_dtypes(include=["int"])
converted_int = data_int.progress_apply(pd.to_numeric, downcast="unsigned")

data_float = data.select_dtypes(include=["float"])
# data_float = data_float.drop(columns="encounterid")
# print(list(data_float))
converted_float = data_float.progress_apply(pd.to_numeric, downcast="float")

data[converted_int.columns] = converted_int
data[converted_float.columns] = converted_float
data[converted_bool.columns] = converted_bool


obj_list = list(data.select_dtypes(include=["object"], exclude=["datetime", "timedelta"]))

for obj in obj_list:
    num_unique_values = len(data[obj].unique())
    num_total_values = len(data[obj])
    if num_unique_values / num_total_values < 0.5:
        data[obj] = data[obj].astype("category")
    

write(config.INTERIM_PARQ, data)
data.to_hdf(config.INTERIM_H5, key='phase_00', mode='a', format='table')

# with h5py.File(h5_file, 'a') as f:
#     try:
#         f.create_dataset('phase_00', data=data)
#     except:
#         print("oops")

# filename1 = config.CLEAN_PHASE_00
# data.to_pickle(filename1)
# print("Clean phase_00 available at:", filename1)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")


##################################################################################


# FOR DEBUGGING
# move this section around wherever you want to save interim results
# file_name = config.INTERIM_DATA_DIR / "data_DEBUG.pickle"
# data.to_pickle(file_name)
# print("Debug data file saved.")


##################################################################################

# OLD STUFF

# print("Fixing dtypes...")
# data_bool = data.select_dtypes(["bool"])
# converted_bool = data_bool * 1.0  # changes bool to float

# print("int to unsigned int...")
# data_int = data.select_dtypes(include=["int"])
# converted_int = data_int.progress_apply(pd.to_numeric, downcast="unsigned")

# data_float = data.select_dtypes(include=["float"])
# converted_float = data_float.progress_apply(pd.to_numeric, downcast="float")

# data[converted_int.columns] = converted_int
# data[converted_float.columns] = converted_float
# data[converted_bool.columns] = converted_bool

# data_obj = data.select_dtypes(
#     include=["object"], exclude=["datetime", "timedelta"]
# ).copy()
# converted_obj = pd.DataFrame()
# for col in data_obj.columns:
#     num_unique_values = len(data_obj[col].unique())
#     num_total_values = len(data_obj[col])
#     if num_unique_values / num_total_values < 0.5:
#         converted_obj.loc[:, col] = data_obj[col].astype("category")
#     else:
#         converted_obj.loc[:, col] = data_obj[col]

# data[converted_obj.columns] = converted_obj


# # Not sure if these next two are necessary anymore
# print("More (probably) useless pd.to_datetime/timedelta commands...")
# data = data.progress_apply(
#     lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
#     axis=0,
# )
# data = data.progress_apply(
#     lambda col: pd.to_timedelta(col, errors="ignore")
#     if (col.dtypes == object)
#     else col,
#     axis=0,
# )
