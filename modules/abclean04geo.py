import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

sys.path.append("modules")
from cbh import config
from cbh import configcols

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

interim_file = config.INTERIM_H5
print("Loading", interim_file)
data = pd.read_hdf(interim_file, key='phase_03')
print("File loaded.")

# data_file = config.CLEAN_PHASE_03
# print("Loading ", data_file)
# data = pd.read_pickle(data_file)

geodata_file = config.GEODATA_FINAL
print("Loading ", geodata_file)
geodata = pd.read_pickle(geodata_file)

print("Replacing govt NaN placeholder with np.nan...")
# govt data uses "-666666666.0" as a NaN placeholder
geodata = geodata.replace(-666666666.0, np.nan)



print("Renaming geodata cols...")
geodata = geodata.rename(
    index=str,
    columns={
        # make the merge easier, match the geodata date col names to the data date col names
        "Admit_Date": "admissiontime_date",
        "Discharge_Date": "dischargetime_date",
        'PatientID': 'patientid',
        # Rename the ACS data so it's interpretable:
        "B01003_001E": "acs_total_population_count",
        "B02001_002E": "acs_race_white_alone",
        "B02001_003E": "acs_race_black_alone",
        "B23025_004E": "acs_population_employed_civilians",
        "B11001_006E": "acs_female_householder_no_husband_present",
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

print("Cleaning newlines...")
geodata['patientid'] = geodata['patientid'].replace(r'\\n','', regex=True)

data = data.drop(columns="patientid_")

# The "for loop" did not generate these as pd datetimes, fix here
date_cols = ["admissiontime_date", "dischargetime_date"]
for col in date_cols:
    data[col] = pd.to_datetime(data[col], yearfirst=True)
    # print(data[col])
    geodata[col] = geodata[col].replace("Not available", np.nan)
    geodata[col] = pd.to_datetime(geodata[col], yearfirst=True)
    # print(geodata[col])

geodata = geodata.sort_values(by=['admissiontime_date'])
data = data.sort_values(by=['admissiontime_date'])

print("Text cleaning for geodata...")
geodata = geodata.progress_apply(
    lambda x: x.str.lower()
    .str.strip()
    .str.replace("\t", "")
    .str.replace("  ", " ")
    .str.replace(" ", "_")
    .str.replace("__", "_")
    if (x.dtype == "object")
    else x
)

geodata = geodata.dropna(subset = ['patientid'])
geodata['patientid'] = geodata['patientid'].astype('category')

result = data.merge(
    geodata, on=["patientid", "admissiontime_date", "dischargetime_date"], how="left"
)

result = result.drop(configcols.USELESS_GEO_COLS, axis=1)

print("Saving data to disk...")
result_file = config.CLEAN_PHASE_04
result.to_pickle(result_file)
result.to_hdf(interim_file, key='phase_04', mode='a', format='table')
print(f"Files available at : {result_file} ; {interim_file}")
# print(result.dtypes)

# Save features to csv
import time

timestrfolder = time.strftime("%Y-%m-%d")
datafolder = config.PROCESSED_DATA_DIR / timestrfolder
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)

feature_list = list(result)
df = pd.DataFrame(feature_list, columns=["features"])

spreadsheet_title = "Feature list 02 geo merged "
timestr = time.strftime("%Y-%m-%d-%H%M")
ext = ".csv"
title = spreadsheet_title + timestr + ext
feature_list_file = datafolder / title

df.to_csv(feature_list_file, index=False)
print("CSV of features available at: ", feature_list_file)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")