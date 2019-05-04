import sys

sys.path.append("modules")

from datetime import datetime

import os
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import numpy as np

tqdm.pandas()

import config

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

pd.options.display.max_columns = 2000

# Load file
filename = config.CLEAN_PHASE_02
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")

# ICU in admission (and ICU LoS?)
icu_date_cols = ["firsticuadmit", "lasticudischarge"]

for col in icu_date_cols:
    data[col] = pd.to_datetime(data[col], yearfirst=True)
    # data[col] = data[col].dt.tz_convert(None)
    colname = f"in_this_admission_{col}"
    print("\n")
    print(colname)
    data[colname] = (
        data.progress_apply(
            lambda row: (row.admissiontime <= row[col] <= row.dischargetime), axis=1
        )
        * 1.0
    )
    # Then, time between admit date and ICU stay start/end
    print(f"time_to_{col}")
    data[f"time_to_{col}"] = data[col] - data["admissiontime"]
    print(f"days_to_{col}")
    data[f"days_to_{col}"] = data.progress_apply(
        lambda row: row[f"time_to_{col}"].total_seconds(), axis=1
    )
    # divide seconds by 3600 to get hours, then 24 to get days
    data[f"days_to_{col}"] = data[f"days_to_{col}"] / 3600
    data[f"days_to_{col}"] = data[f"days_to_{col}"] / 24
    # went to ICU within 24h of admission?
    print(f"on_day_of_admission_{col}")
    data[f"on_day_of_admission_{col}"] = (
        data[f"days_to_{col}"] < 1.5
    ) & (  # 1.5 gives a little fudge factor for admit times and notes
        data[f"days_to_{col}"] > 0
    )
    data[f"on_day_of_admission_{col}"] = (
        data[f"on_day_of_admission_{col}"] * 1.0
    )  # convert to 1/0 instead of true/false
    ###TODO: Consider something like an "ICU ever prior to admission" binary col

print("Generating ICU  length of stay...")
# data["lasticudischarge"] = pd.to_datetime(data["lasticudischarge"])
# data["firsticuadmit"] = pd.to_datetime(data["firsticuadmit"])
data["icu_length_of_stay_in_time"] = data["lasticudischarge"] - data["firsticuadmit"]
data["icu_length_of_stay_in_days"] = data.progress_apply(
    lambda row: row.icu_length_of_stay_in_time.total_seconds(), axis=1
)
data["icu_length_of_stay_in_days"] = data["icu_length_of_stay_in_days"] / 3600
data["icu_length_of_stay_in_days"] = data["icu_length_of_stay_in_days"] / 24
# drop negative ICU LoS
a = np.array(data["icu_length_of_stay_in_days"].values.tolist())
data["icu_length_of_stay_in_days"] = np.where(a < 0, np.nan, a).tolist()
# drop really long ICU LoS
a = np.array(data["icu_length_of_stay_in_days"].values.tolist())
data["icu_length_of_stay_in_days"] = np.where(a > 100, np.nan, a).tolist()

# some people have an ICU LoS longer than their normnal LoS.
# for most, the discrepancy is less than a day
# drop those for whom the discrepancy is greater than 2 days
print("Dropping ICU LoS > LoS...")
print(len(data))
data["icuminuslos"] = data["icu_length_of_stay_in_days"] - data["length_of_stay_in_days"]
a = np.array(data["icuminuslos"].values.tolist())
data["icuminuslos"] = np.where(a > 2, "weird", a).tolist()
data = data[data.icuminuslos != "weird"]
print(len(data))

data = data.sort_index(axis=1) # get all the cols in alphabetical order

print("Saving to file...")
filename1 = config.CLEAN_PHASE_03
data.to_pickle(filename1)
print("Clean phase_03 available at:", filename1)

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


# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
