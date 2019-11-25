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

data_file = config.CLEAN_PHASE_03
print("Loading ", data_file)
data = pd.read_pickle(data_file)

geodata_file = config.GEODATA_FINAL
print("Loading ", geodata_file)
geodata = pd.read_pickle(geodata_file)

print("Replacing govt NaN placeholder with np.nan...")
# govt data uses "-666666666.0" as a NaN placeholder
geodata = geodata.replace(-666666666.0, np.nan)


# make the merge easier, match the geodata date col names to the data date col names
print("Renaming geodata cols...")
geodata = geodata.rename(
    index=str,
    columns={
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

print("Merging...")

data = data.drop(columns="patientid_")

# The "for loop" did not generate these as pd datetimes, fix here
date_cols = ["admissiontime_date", "dischargetime_date"]
# print(list(data))

for col in date_cols:
    data[col] = pd.to_datetime(data[col], yearfirst=True)
    # print(data[col])
    geodata[col] = geodata[col].replace("Not available", np.nan)
    geodata[col] = pd.to_datetime(geodata[col], yearfirst=True)
    # print(geodata[col])
    
result = data.merge(
    geodata, on=["patientid", "admissiontime_date", "dischargetime_date"], how="left"
)
result = result.drop(configcols.USELESS_GEO_COLS, axis=1)

result_file = config.CLEAN_PHASE_04
result.to_pickle(result_file)
print("File available at :", result_file)
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
timestrfolder = time.strftime("%Y-%m-%d")
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
