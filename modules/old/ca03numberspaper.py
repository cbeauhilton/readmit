#%%
import os
import sys

sys.path.append("modules")
from datetime import datetime

import numpy as np
import pandas as pd
from plotly.offline import iplot
from tableone import TableOne
import time

from cbh import config

# print("About to run ", os.path.basename(__file__))
startTime = datetime.now()


# display all columns when viewing dataframes: make the number
# anything bigger than your number of columns
pd.options.display.max_columns = 2000


figures_path = config.FIGURES_DIR
tables_path = config.TABLES_DIR
filename = config.PROCESSED_FINAL_DESCRIPTIVE
print("Loading data file: ", filename)
data = pd.read_pickle(filename)
print("Loaded.")
print(list(data))
#%%



dummy = 1
n_pts = data["patientid"].nunique()
n_encs = data["encounterid"].nunique()
n_30dreadmits = data["readmitted30d"].sum()
readmit_rate = n_30dreadmits / n_encs

data_over_65 = data[data.patient_age >= 65].copy()
n_encs_over_65 = data_over_65["encounterid"].nunique()
n_30dreadmits_over_65 = data_over_65["readmitted30d"].sum()
readmit_rate_over_65 = n_30dreadmits_over_65 / n_encs_over_65

age_medians = data.groupby(["readmitted30d"])["patient_age"].median()
age_median_nonreadmit = age_medians.iloc[0]
age_median_readmit = age_medians.iloc[1]

blackdf = data[data["race"] == "Black"]
blackdf = blackdf[["race", "readmitted30d"]]
black_readmission = len(blackdf[blackdf["readmitted30d"] == 1]) / len(blackdf)

whitedf = data[data["race"] == "White"]
whitedf = whitedf[["race", "readmitted30d"]]
white_readmission = len(whitedf[whitedf["readmitted30d"] == 1]) / len(whitedf)

relationship = data.groupby(["readmitted30d"])

marrieddf = data[data["marital_status"] == "Married or partnered"]
marrieddf = marrieddf[["marital_status", "readmitted30d"]]
married_readmission = len(marrieddf[marrieddf["readmitted30d"] == 1]) / len(marrieddf)

divorceddf = data[data["marital_status"] == "Divorced or separated"]
divorceddf = divorceddf[["marital_status", "readmitted30d"]]
divorced_readmission = len(divorceddf[divorceddf["readmitted30d"] == 1]) / len(
    divorceddf
)

widoweddf = data[data["marital_status"] == "Widowed"]
widoweddf = widoweddf[["marital_status", "readmitted30d"]]
widowed_readmission = len(widoweddf[widoweddf["readmitted30d"] == 1]) / len(widoweddf)

singledf = data[data["marital_status"] == "Single"]
singledf = singledf[["marital_status", "readmitted30d"]]
single_readmission = len(singledf[singledf["readmitted30d"] == 1]) / len(singledf)

medicare = data[data["financialclass_Medicare"] == 1]
medicare = medicare[["financialclass_Medicare", "readmitted30d"]]
medicare_readmission = len(medicare[medicare["readmitted30d"] == 1]) / len(medicare)

private = data[data["financialclass_Private_Health_Insurance"] == 1]
private = private[["financialclass_Private_Health_Insurance", "readmitted30d"]]
private_readmission = len(private[private["readmitted30d"] == 1]) / len(private)


medicare = data.groupby(["readmitted30d"])

min_los_per_pt = data.length_of_stay_in_days.min()
mean_los_per_pt = data.length_of_stay_in_days.mean()
q1_los_per_pt = data.length_of_stay_in_days.quantile(0.25)
median_los_per_pt = data.length_of_stay_in_days.median()
q3_los_per_pt = data.length_of_stay_in_days.quantile(0.75)
max_los_per_pt = data.length_of_stay_in_days.max()

noobs = data.copy()
noobs = noobs[noobs["patientclassdescription"] != "Observation"]
noobs_min_los_per_pt = noobs.length_of_stay_in_days.min()
noobs_mean_los_per_pt = noobs.length_of_stay_in_days.mean()
noobs_q1_los_per_pt = noobs.length_of_stay_in_days.quantile(0.25)
noobs_median_los_per_pt = noobs.length_of_stay_in_days.median()
noobs_q3_los_per_pt = noobs.length_of_stay_in_days.quantile(0.75)
noobs_max_los_per_pt = noobs.length_of_stay_in_days.max()
print(noobs_median_los_per_pt)
print(noobs_mean_los_per_pt)
#%%
nobabs = data.copy()
print(len(nobabs))

babs_list = [
    "Single liveborn infant delivered vaginally",
    "Single lbby c section",
    "Single liveborn infant delivered by cesarean",
    "Prev c sect nos deliver",
    "Post term preg delivered" ,
    "Post term pregnancy" ,
    "Del w 1 deg lacerat del" ,
    "Oth curr cond delivered" ,
    "Abn fetal hrt rate del" ,
    "Elderly multigravida del" ,
    "Breech presentat deliver" ,
    "Thrt prem labor antepart" ,
    "Twin pregnancy delivered",
    "High head at term deliv" ,
    "Twin liveborn infant delivered vaginally",
    "Early onset delivery del",
    "Poor fetal growth deliv",
]

for bybybab in babs_list:
        nobabs = nobabs[nobabs["primary_diagnosis_code"] != bybybab]
        print(bybybab)
        print(len(nobabs))
#%%

# print(len(nobabs))
print(f"Num deliveries dropped: {len(data) - len(nobabs)}")
nobabs = nobabs[nobabs["patientclassdescription"] != "Observation"]
# ~10,000 outpt
nobabs = nobabs[nobabs["patientclassdescription"] != "Outpatient"]
# ~8,000 amb surg
nobabs = nobabs[nobabs["patientclassdescription"] != "Ambulatory Surgical Procedures"]  
# ~7,000
nobabs = nobabs[nobabs["patientclassdescription"] != "Emergency"]
print(len(nobabs))
print(f"Num deliveries, obs, outpt, abm surg, emergency dropped: {len(data) - len(nobabs)}")
#%%
# pd.options.display.max_rows = 2000
# print(nobabs["primary_diagnosis_code"].value_counts().to_frame())
#%%
nobabs_min_los_per_pt = nobabs.length_of_stay_in_days.min()
nobabs_mean_los_per_pt = nobabs.length_of_stay_in_days.mean()
nobabs_q1_los_per_pt = nobabs.length_of_stay_in_days.quantile(0.25)
nobabs_median_los_per_pt = nobabs.length_of_stay_in_days.median()
nobabs_q3_los_per_pt = nobabs.length_of_stay_in_days.quantile(0.75)
nobabs_max_los_per_pt = nobabs.length_of_stay_in_days.max()
print(nobabs_median_los_per_pt)
print(nobabs_mean_los_per_pt)
#%%


#%%

data0 = data.copy()
data0.length_of_stay_in_days = data0.length_of_stay_in_days.apply(
    np.floor
)  # round down
mode_los_per_pt = data0.length_of_stay_in_days.mode()

pt_enc = data[["patientid", "encounterid"]]
df = (
    pt_enc.groupby("patientid")["encounterid"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="encounternum")
)

earliest_admit = min(data["admissiontime"])
latest_admit = max(data["admissiontime"])
earliest_discharge = min(data["dischargetime"])
latest_discharge = max(data["dischargetime"])

min_enc_per_pt = df.encounternum.min()
mean_enc_per_pt = df.encounternum.mean()
q1_enc_per_pt = df.encounternum.quantile(0.25)
median_enc_per_pt = df.encounternum.median()
q3_enc_per_pt = df.encounternum.quantile(0.75)
max_enc_per_pt = df.encounternum.max()
idx_pt_w_max_encs = df.encounternum.idxmax()

pts_one_encounter = df[df["encounternum"] == 1].nunique()
pts_one_encounter = pts_one_encounter.patientid

# df.encounternum.hist(bins=177)
abs_median_devs = abs(df.encounternum - df.encounternum.median())
abs_median_devs = abs_median_devs.median() * 1.4826
# print(abs_median_devs)
timestr = time.strftime("%Y-%m-%d-%H%M")
# df.boxplot(column="encounternum", return_type="axes", figsize=(8, 20))

d = [
    [
        timestr,
        earliest_admit,
        latest_admit,
        earliest_discharge,
        latest_discharge,
        n_pts,
        n_encs,
        n_30dreadmits,
        readmit_rate,
        n_encs_over_65,
        n_30dreadmits_over_65,
        mean_enc_per_pt,
        min_enc_per_pt,
        q1_enc_per_pt,
        median_enc_per_pt,
        noobs_median_los_per_pt,
        nobabs_median_los_per_pt,
        q3_enc_per_pt,
        max_enc_per_pt,
        idx_pt_w_max_encs,
        pts_one_encounter,
        mean_los_per_pt,
        min_los_per_pt,
        q1_los_per_pt,
        median_los_per_pt,
        q3_los_per_pt,
        max_los_per_pt,
        mode_los_per_pt,
    ]
]

csv_df = pd.DataFrame(
    d,
    columns=(
        "Date",
        "First Admission Date",
        "Last Admission Date",
        "First Discharge Date",
        "Last Discharge Date",
        "Number of Unique Patients",
        "Number of Encounters",
        "Number of 30 Day Readmissions",
        "Number of Encounters for Patients 65 or Older",
        "Number of 30 Day Readmissions for Patients 65 or Older",
        "Readmission Rate",
        "Mean Encounters per Patient",
        "Minimum Encounters per Patient",
        "Q1 Encounters per Patient",
        "Median Encounters per Patient",
        "Median Encounters per Patient Excluding Observation",
        "Median Encounters per Patient Excluding Observation and Labor and Delivery",
        "Q3 Encounters per Patient",
        "Max Encounters per Patient",
        "Index of Patient with Max Encounters",
        "Number of Patients with Only One Encounter",
        "Mean Length of Stay per Patient",
        "Minimum Length of Stay per Patient",
        "Q1 Length of Stay per Patient",
        "Median Length of Stay per Patient",
        "Q3 Length of Stay per Patient",
        "Max Length of Stay per Patient",
        "Mode Length of Stay per Patient",
    ),
)

# print(csv_df.head())
print("Making csv with important numbers...")
csv_df.to_csv(config.PAPER_NUMBERS, mode="w", header=True)

# Note on formatting:
# The colon allows you to specify options, the comma adds commas, the ".2f" says how many decimal points to keep.
# Nice tutorial here: https://stackabuse.com/formatting-strings-with-python/

sentence01 = f"In the study period there were {n_encs:,} hospitalizations for {n_pts:,} unique patients, {pts_one_encounter:,} ({pts_one_encounter/n_pts*100:.0f}%) of whom had only one hospitalization recorded. "
sentence02 = f"The median number of hospitalizations per patient was {median_enc_per_pt:.0f} (range {min_enc_per_pt:.0f}-{max_enc_per_pt:.0f}, [{q1_enc_per_pt} , {q3_enc_per_pt}]). "
sentence03 = f"There were {n_30dreadmits:,} thirty-day readmissions for an overall readmission rate of {readmit_rate*100:.0f}%. "
sentence03a = f"Among patients aged 65 years or older, the thirty-day readmission rate was {readmit_rate_over_65*100:.0f}%. "
sentence04 = f"The median LOS, including patients in observation status and labor and delivery patients, was {median_los_per_pt:,.2f} days (range {min_los_per_pt:.0f}-{max_los_per_pt:.0f}, [{q1_los_per_pt} , {q3_los_per_pt}]). "
sentence04a = f"The median LOS excluding patients in observation status was {noobs_median_los_per_pt:,.2f} days (range {noobs_min_los_per_pt:.0f}-{noobs_max_los_per_pt:.0f}, [{noobs_q1_los_per_pt} , {noobs_q3_los_per_pt}]). "
sentence04b = f"The median LOS excluding patients in observation status and labor and delivery patients, was {nobabs_median_los_per_pt:,.2f} days (range {nobabs_min_los_per_pt:.0f}-{nobabs_max_los_per_pt:.0f}, [{nobabs_q1_los_per_pt} , {nobabs_q3_los_per_pt}]). "
sentence05 = f"The demographic and clinical characteristics of the patient cohort are summarized in Table 1. "
sentence06a = f"Higher rates of 30-day readmissions were observed in patients who were older (median age {age_median_readmit:.0f} vs. {age_median_nonreadmit:.0f} years), "
sentence06b = f"African American (rate of {black_readmission*100:.0f}% vs. {white_readmission*100:.0f}% in whites), "
# sentence06c = f"divorced/separated or widowed (rates of {divorced_readmission*100:.0f}%, {widowed_readmission*100:.0f}% vs. rates of {married_readmission*100:.0f}%, {single_readmission*100:.0f}% for married/partnered or single patients, respectively), "
sentence06c = f"divorced/separated or widowed (rates of {divorced_readmission*100:.0f}% vs. rates of {married_readmission*100:.0f}% for married/partnered or single patients), "
sentence06d = f"on Medicare insurance (rate of {medicare_readmission*100:.0f}% vs. {private_readmission*100:.0f}% for private insurance), "
sentence06e = "and had one or multiple chronic conditions such as cancer, renal disease, congestive heart failure, and chronic obstructive pulmonary disease, etc. (\hyperref[table:table1]{Table 1})."
paragraph01 = (
    sentence01
    + sentence02
    + sentence03
    + sentence03a
    + sentence04
    + sentence04a
    + sentence04b
    + sentence05
    + sentence06a
    + sentence06b
    + sentence06c
    + sentence06d
    + sentence06e
)

# Print paragraph to the terminal...
print(paragraph01)

# Define file...
if not os.path.exists(config.TEXT_DIR):
    print("Making folder called", config.TEXT_DIR)
    os.makedirs(config.TEXT_DIR)

results_text_file = config.TEXT_DIR / "results_paragraphs.txt"
# ...and save.
with open(results_text_file, "w") as text_file:
    print(paragraph01, file=text_file)


text_file_latex = config.TEXT_DIR / "results_paragraphs_latex.txt"
# and make a LaTeX-friendly version (escape the % symbols with \)
# Read in the file
with open(results_text_file, "r") as file:
    filedata = file.read()
# Replace the target string
filedata = filedata.replace("%", "\%")
# Write the file
with open(text_file_latex, "w") as file:
    file.write(filedata)

results_clf_df = pd.read_csv(config.TRAINING_REPORTS)
results_reg_df = pd.read_csv(config.REGRESSOR_TRAINING_REPORTS)

# I wrote the CSV loop so it adds a new header every time it writes something new,
# which is pretty useful but not here. Drop all rows with the extra header info.
results_clf_df = results_clf_df[results_clf_df.Time != "Time"]
# make the Time column a pandas datetime column

try:
    results_clf_df["Time"] = pd.to_datetime(results_clf_df["Time"], format="%Y-%m-%d-%H%M")
except:
    results_clf_df['Time'] = pd.to_datetime(results_clf_df['Time'], format="_%Y-%m-%d-%H%M_")

results_clf_df = results_clf_df[["Time", "Target", "ROC AUC", "Brier Score Loss"]].copy()
print(results_clf_df)
print(results_clf_df[["Time", "Target", "ROC AUC", "Brier Score Loss"]].head())
results_clf_df["ROC AUC"] = pd.to_numeric(results_clf_df["ROC AUC"])
results_clf_df["Brier Score Loss"] = pd.to_numeric(results_clf_df["Brier Score Loss"]) 
# # select the run with the full number of features?
# results_clf_df = results_clf_df.loc[results_clf_df.groupby('Target')["Number of Features"]idxmax()]
# # select the newest training report for each target
# results_clf_df = results_clf_df.loc[results_clf_df.groupby('Target').Time.idxmax()]
# select the run with highest AUC
results_clf_df = results_clf_df.loc[
    results_clf_df.groupby("Target")["ROC AUC"].idxmax()
]
print(results_clf_df)

with open(config.RESULTS_TEX, "w") as tf:
    tf.write(results_clf_df.to_latex())

print("Done.")
#%%
# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")


"""

"""
