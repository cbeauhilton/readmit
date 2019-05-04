import sys

sys.path.append("modules")

from datetime import datetime

import os
import pandas as pd
import lightgbm as lgb

import config

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

pd.options.display.max_columns = 2000

# Load file
filename = config.CLEAN_PHASE_09
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")

data.hemoglobin_discharge_value = pd.to_numeric(data.hemoglobin_discharge_value, errors='coerce')
data["Low hemoglobin level (<12) at discharge"] = data['hemoglobin_discharge_value'].apply(lambda x: 1 if x<12 else 0)

# print(data.sodium_discharge_value.value_counts())
data.sodium_discharge_value = pd.to_numeric(data.sodium_discharge_value, errors='coerce')
data["Low sodium level (<135) at discharge"] = data['sodium_discharge_value'].apply(lambda x: 1 if x<135 else 0)

print(data["length_of_stay_in_days"].describe())

print(len(data))
data = data.drop_duplicates(subset=["encounterid"], keep="first")
print(len(data))

# Save features to csv
import time
timestrfolder = time.strftime("%Y-%m-%d")
datafolder = config.PROCESSED_DATA_DIR / timestrfolder
if not os.path.exists(datafolder):
    print("Making folder called", datafolder)
    os.makedirs(datafolder)
feature_list = list(data)
df = pd.DataFrame(feature_list, columns=["features"])
spreadsheet_title = "Feature list 04 TableOne "
timestr = time.strftime("%Y-%m-%d-%H%M")
timestrfolder = time.strftime("%Y-%m-%d")
ext = ".csv"
title = spreadsheet_title + timestr + ext
feature_list_file = datafolder / title
df.to_csv(feature_list_file, index=False)
print("CSV of features available at: ", feature_list_file)


print("Saving to file...")
filename1 = config.CLEAN_PHASE_11_TABLEONE
data.to_pickle(filename1)
print("Clean phase_00_tableone available at:", filename1)

# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
