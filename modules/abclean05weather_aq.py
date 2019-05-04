"""
###TODO: Make this happen. Now is just copy of geo data merge
"""

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
import config
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
print("Renaming geodata date cols...")
geodata = geodata.rename(
    index=str,
    columns={
        "Admit_Date": "admissiontime_date", 
        "Discharge_Date": "dischargetime_date",
    },
)

print("Merging...")
# The "for loop" did not generate these as pd datetimes, fix here
date_cols = ["admissiontime_date", "dischargetime_date"]
for col in date_cols:
    data[col] = pd.to_datetime(data[col], yearfirst=True)
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
spreadsheet_title = "Feature list aq merged "
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
