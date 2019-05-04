#%%
import os
import sys

sys.path.append("modules")
from datetime import datetime

import pandas as pd
from plotly.offline import iplot
from tableone import TableOne
import time
import config


print("About to run ", os.path.basename(__file__))
startTime = datetime.now()


# display all columns when viewing dataframes: make the number
# anything bigger than your number of columns
pd.options.display.max_columns = 2000

figures_path = config.FIGURES_DIR
tables_path = config.TABLES_DIR
filename = config.PROCESSED_FINAL_DESCRIPTIVE
# print("Loading data file: ", filename)
# data = pd.read_pickle(filename)
# print("Loaded.")
# print(list(data))
#%%

import numpy as np

results_clf_df = pd.read_csv(config.TRAINING_REPORTS)
results_reg_df = pd.read_csv(config.REGRESSOR_TRAINING_REPORTS)

# I wrote the CSV loop so it adds a new header every time it writes something new,
# which is pretty useful but not here. Drop all rows with the extra header info.
results_clf_df = results_clf_df[results_clf_df.Time != "Time"]
# make the Time column a pandas datetime column
results_clf_df["Time"] = pd.to_datetime(results_clf_df["Time"], format="%Y-%m-%d-%H%M")
results_clf_df = results_clf_df[
    ["Time", "Target", "ROC AUC", "Brier Score Loss", "Average Precision"]
]
# print(results_clf_df)
# print(results_clf_df[["Time", "Target", "ROC AUC"]].head())
results_clf_df["ROC AUC"] = pd.to_numeric(results_clf_df["ROC AUC"])
results_clf_df["Brier Score Loss"] = pd.to_numeric(results_clf_df["Brier Score Loss"])
results_clf_df["Average Precision"] = pd.to_numeric(results_clf_df["Average Precision"])
results_clf_df = results_clf_df.round(2)
# # select the run with the full number of features?
# results_clf_df = results_clf_df.loc[results_clf_df.groupby('Target')["Number of Features"]idxmax()]
# # select the newest training report for each target
# results_clf_df = results_clf_df.loc[results_clf_df.groupby('Target').Time.idxmax()]
# select the run with highest AUC
results_clf_df = results_clf_df.loc[
    results_clf_df.groupby("Target")["ROC AUC"].idxmax()
]
# results_clf_df = results_clf_df.set_index("Target")
print(results_clf_df)

# I wrote the CSV loop so it adds a new header every time it writes something new,
# which is pretty useful but not here. Drop all rows with the extra header info.
results_reg_df = results_reg_df[results_reg_df.Time != "Time"]
# make the Time column a pandas datetime column
results_reg_df["Time"] = pd.to_datetime(results_reg_df["Time"], format="%Y-%m-%d-%H%M")
results_reg_df = results_reg_df[["Time", "Target", "RMSE"]]
# print(results_clf_df)
# print(results_clf_df[["Time", "Target", "ROC AUC"]].head())
results_reg_df["RMSE"] = pd.to_numeric(results_reg_df["RMSE"])
results_reg_df = results_reg_df.round(2)
# # select the run with the full number of features?
# results_clf_df = results_clf_df.loc[results_clf_df.groupby('Target')["Number of Features"]idxmax()]
# # select the newest training report for each target
# results_clf_df = results_clf_df.loc[results_clf_df.groupby('Target').Time.idxmax()]
# select the run with highest AUC
results_reg_df = results_reg_df.loc[results_reg_df.groupby("Target")["RMSE"].idxmin()]
# results_reg_df = results_reg_df.set_index("Target")
print(results_reg_df)
all_results = pd.concat([results_clf_df, results_reg_df], axis=0, ignore_index=True)
all_results = all_results[
    ["Target", "ROC AUC", "Brier Score Loss", "Average Precision", "RMSE"]
]
all_results = all_results.reset_index(drop=True)
all_results["Target"].replace(
    {
        "financialclass_binary": "Financial Class",
        "gender_binary": "Gender",
        "length_of_stay_over_3_days": "Hospital stay over 3 days",
        "length_of_stay_over_5_days": "Hospital stay over 5 days",
        "length_of_stay_over_7_days": "Hospital stay over 7 days",
        "race_binary": "Race",
        "readmitted30d": "Readmitted within 30 days",
        "readmitted3d": "Readmitted within 3 days",
        "readmitted5d": "Readmitted within 5 days",
        "readmitted7d": "Readmitted within 7 days",
        "days_between_current_discharge_and_next_admission": "Days to readmission",
        "length_of_stay_in_days": "Length of stay",
        "patient_age": "Age",
    },
    inplace=True,
)
all_results = all_results.set_index("Target")
reindex_list = [
    "Readmitted within 30 days",
    "Readmitted within 7 days",
    "Readmitted within 5 days",
    "Readmitted within 3 days",
    "Days to readmission",
    "Hospital stay over 7 days",
    "Hospital stay over 5 days",
    "Hospital stay over 3 days",
    "Length of stay",
    "Gender",
    "Race",
    "Financial Class",
    "Age",
]
all_results = all_results.reindex(reindex_list)
all_results = all_results.fillna(value = "--")
print(all_results)
with open(config.RESULTS_TEX, "w") as tf:
    tf.write(all_results.to_latex())

print("Done.")
#%%
# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")


"""

"""
