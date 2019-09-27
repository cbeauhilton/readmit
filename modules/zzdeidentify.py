import hashlib
import os
import random
import string
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import cbh.config as config
import configcols

tqdm.pandas()

try:
    import cPickle as pickle
except BaseException:
    import pickle

print("About to run", os.path.basename(__file__))
startTime = datetime.now()


###############################################################################
#     __                __
#    / /___  ____ _____/ /
#   / / __ \/ __ `/ __  /
#  / / /_/ / /_/ / /_/ /
# /_/\____/\__,_/\__,_/
#
###############################################################################


unscrubbed_file = config.UNSCRUBBED_H5

# filename = config.CLEAN_PHASE_09
# print("Loading", filename)
# data = pd.read_pickle(filename)

# data = data.progress_apply(
#     lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
#     axis=0,
# )
# data.to_hdf(unscrubbed_file, key="unscrubbed", mode="w", format="table")
print("Loading", unscrubbed_file)
data = pd.read_hdf(unscrubbed_file, key="unscrubbed")

# print(len(list(data)))
# print(sorted(list(data)))


###############################################################################
#        __     _____
#   ____/ /__  / __(_)___  ___
#  / __  / _ \/ /_/ / __ \/ _ \
# / /_/ /  __/ __/ / / / /  __/
# \__,_/\___/_/ /_/_/ /_/\___/
#
###############################################################################


dropcols = [
    "admissiontime_date",
    "admissiontime_month",
    "admissiontime_weekofyear",
    "admissiontime_within_24h",
    "admissiontime_year",
    "admissiontime",
    "cbc_admit_time_date",
    "cbc_admit_time_month",
    "cbc_admit_time_weekofyear",
    "cbc_admit_time_year",
    "cbc_admit_time",
    "cbc_discharge_time_date",
    "cbc_discharge_time_month",
    "cbc_discharge_time_weekofyear",
    "cbc_discharge_time_within_24h",
    "cbc_discharge_time_year",
    "cbc_discharge_time",
    "cmp_admit_time_date",
    "cmp_admit_time_month",
    "cmp_admit_time_weekofyear",
    "cmp_admit_time_year",
    "cmp_admit_time",
    "cmp_discharge_time_date",
    "cmp_discharge_time_month",
    "cmp_discharge_time_quarter",
    "cmp_discharge_time_weekofyear",
    "cmp_discharge_time_year",
    "cmp_discharge_time",
    "dischargetime_date",
    "dischargetime_month",
    "dischargetime_weekofyear",
    "dischargetime_year",
    "dischargetime",
    "encounterid",
    "epicdeathdate",
    "firsticuadmit",
    "lasticudischarge",
    "mrn",
    "ohiodeathindexdate",
]

docols = [
    "patient_age",  # round to nearest 5 and 10
    "patientid",  # convert to hash code
    "newname",  # generated in patientid conversion
    "time_between_current_admission_and_previous_discharge",  # converted to days with fractions
    "time_between_current_discharge_and_next_admission",  # converted to days with fractions
    "time_from_admission_to_epicdeathdate",  # converted to days with fractions
    "time_from_admission_to_ohiodeathindexdate",  # converted to days with fractions
    "time_from_discharge_to_epicdeathdate",  # converted to days with fractions
    "time_from_discharge_to_ohiodeathindexdate",  # converted to days with fractions
    "time_since_beginning_of_last_admission",  # converted to days with fractions
    "time_to_firsticuadmit",  # converted to days with fractions
    "time_to_lasticudischarge",  # converted to days with fractions
    "lastvisitdate", # converted to days from other important dates
]

# for colum in docols:
    # print(data[colum])
    # print("\n")


###############################################################################
#                          __                _
#    _________ _____  ____/ /___  ____ ___  (_)___  ___
#   / ___/ __ `/ __ \/ __  / __ \/ __ `__ \/ /_  / / _ \
#  / /  / /_/ / / / / /_/ / /_/ / / / / / / / / /_/  __/
# /_/   \__,_/_/ /_/\__,_/\____/_/ /_/ /_/_/ /___/\___/
#
###############################################################################


# randomize the patientid, but keep it consistent (so each patientid has a unique code)
# https://medium.com/luckspark/hashing-pandas-dataframe-column-with-nonce-763a8c23a833
# Get a unique list of the clear text, as a List
tmplist = list(set(data["patientid"]))
# Add some random characters before and after the patientid
# Structure them in a Dictionary
# Example -- Liverpool -> aaaaaaaLiverpoolbbbbbbbb
mapping1 = {
    i: ("".join(random.choice(string.hexdigits) for i in range(12)))
    + i
    + ("".join(random.choice(string.hexdigits) for i in range(12)))
    for i in tmplist
}
# Create a DF to leave the original DF intact.
df = data.copy()
# Create a new column containing clear_text_Nonce
df["newname"] = [mapping1[i] for i in df["patientid"]]
# Hash the clear_text+Nonce string
df["pt_id_hash"] = [hashlib.sha1(str.encode(str(i))).hexdigest() for i in df["newname"]]
# print(df[["patientid", "pt_id_hash", "newname"]].head())


###############################################################################
#         __
#   _____/ /__  ____ _____
#  / ___/ / _ \/ __ `/ __ \
# / /__/ /  __/ /_/ / / / /
# \___/_/\___/\__,_/_/ /_/
#
###############################################################################


class renamer():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])

df = df.rename(columns=renamer())
print(sorted(list(df)))


###############################################################################
#                                __
#    _________  __  ______  ____/ /
#   / ___/ __ \/ / / / __ \/ __  /
#  / /  / /_/ / /_/ / / / / /_/ /
# /_/   \____/\__,_/_/ /_/\__,_/
#
###############################################################################


# round patient_age to nearest 5 and 10
def custom_round(x, base=5):
    return int(base * round(float(x) / base))


df["patient_age_nearest_5"] = df.patient_age.apply(lambda x: custom_round(x, base=5))
df["patient_age_nearest_10"] = df.patient_age.round(-1)
# print(df[["patient_age", "patient_age_nearest_5", "patient_age_nearest_10"]])


###############################################################################
#                                    __
#   _________  ____ _   _____  _____/ /_
#  / ___/ __ \/ __ \ | / / _ \/ ___/ __/
# / /__/ /_/ / / / / |/ /  __/ /  / /_
# \___/\____/_/ /_/|___/\___/_/   \__/
#
###############################################################################


timed_cols = [
    # "time_between_current_admission_and_previous_discharge",
    # "time_between_current_discharge_and_next_admission",
    "time_from_admission_to_epicdeathdate",
    "time_from_admission_to_ohiodeathindexdate",
    "time_from_discharge_to_epicdeathdate",
    "time_from_discharge_to_ohiodeathindexdate",
    "time_since_beginning_of_last_admission",
    "time_to_firsticuadmit",
    "time_to_lasticudischarge",
]

for colum in timed_cols:
    df[f"{colum}_in_days"] = df[colum].dt.total_seconds()
    # Multiply by negative 1 to get a positive number
    # df[colum] = df[colum] * -1
    df[f"{colum}_in_days"] = df[f"{colum}_in_days"] / 3600
    df[f"{colum}_in_days"] = df[f"{colum}_in_days"] / 24

# print(df[timed_cols].describe())


###############################################################################
#               __           __      __
#   _________ _/ /______  __/ /___ _/ /____
#  / ___/ __ `/ / ___/ / / / / __ `/ __/ _ \
# / /__/ /_/ / / /__/ /_/ / / /_/ / /_/  __/
# \___/\__,_/_/\___/\__,_/_/\__,_/\__/\___/
#
###############################################################################


time_cal_cols = [
    "dischargetime",
    "ohiodeathindexdate",
    "epicdeathdate",
    "admissiontime",
    "firsticuadmit",
    "lasticudischarge",
]

for col in time_cal_cols:
    df[col] = pd.to_datetime(df[col])
    timecolname = f"time_{col}_minus_last_visit_date"
    print(timecolname)
    df[timecolname] = df[col] - df["lastvisitdate"]
    hourscolname = f"days_{col}_minus_last_visit_date"
    print(hourscolname)
    df[hourscolname] = df.progress_apply(
        lambda row: row[timecolname].total_seconds(), axis=1
    )
    df[hourscolname] = df[hourscolname] / 3600
    df[hourscolname] = df[hourscolname] / 24
    # print(df[[hourscolname, timecolname]])

df["died_on_record_0"] = np.where(df["epicdeathdate"].isnull(), "", "dead")
df["died_on_record_1"] = np.where(df["ohiodeathindexdate"].isnull(), "", "dead")
df['died_on_record'] = df['died_on_record_1'].fillna('') + df['died_on_record_0'].fillna('')
# replace field that's entirely space (or empty) with NaN
df = df.replace(r'^\s*$', "no_death_recorded", regex=True)
df = df.replace('deaddead', "dead")
df = df.drop(["died_on_record_1", "died_on_record_0"], axis=1)


###############################################################################
#                          __
#    _________________  __/ /_
#   / ___/ ___/ ___/ / / / __ \
#  (__  ) /__/ /  / /_/ / /_/ /
# /____/\___/_/   \__,_/_.___/
#
###############################################################################


print(len(df))
df = df.drop(dropcols, axis=1)
df = df.drop(docols, axis=1)
print(len(df))


###############################################################################
#    _________ __   _____
#   / ___/ __ `/ | / / _ \
#  (__  ) /_/ /| |/ /  __/
# /____/\__,_/ |___/\___/
#
###############################################################################


unscrubbed_file = config.UNSCRUBBED_H5
scrubbed_file = config.SCRUBBED_H5

df.to_hdf(unscrubbed_file, key="scrubbed", mode="a", format="table")
df.to_hdf(scrubbed_file, key="scrubbed", mode="a", format="table")


###############################################################################
#                   __
#   ___  ____  ____/ /
#  / _ \/ __ \/ __  /
# /  __/ / / / /_/ /
# \___/_/ /_/\__,_/
#
###############################################################################


# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
