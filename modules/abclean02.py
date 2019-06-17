import sys

sys.path.append("modules")

from datetime import datetime


import os
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

tqdm.pandas()

from cbh import config
import configcols

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

pd.options.display.max_columns = 2000

# Load file
filename = config.CLEAN_PHASE_01
print("Loading", filename)
data = pd.read_pickle(filename)
print("File loaded.")


print("Fixing some ugly column names...")
data = data.rename(
    index=str,
    columns={
        "readmitted0.5d": "readmitted0_5d",
        "financialclass_Private Health Insurance": "financialclass_Private_Health_Insurance",
        "pressureulcer_Present on Admission to the Hospital": "pressureulcer_Present_on_Admission_to_the_Hospital",
        "pressureulcer_History of Pressure Ulcer": "pressureulcer_History_of_Pressure_Ulcer",
    },
)

data["length_of_stay_of_last_admission"] = (
    data["length_of_stay_of_last_admission"] * -1
)  # fix the sign on this column
data = data.drop(
    ["length_of_stay"], axis=1
)  # old LoS column no longer needed, the new one is better

# data.bmi_admit = pd.to_numeric(data.bmi_admit, errors='coerce')
# data.bmi_discharge = pd.to_numeric(data.bmi_discharge, errors='coerce')
# data.calcium_admit_value = pd.to_numeric(data.calcium_admit_value, errors='coerce')
# data.calcium_discharge_value = pd.to_numeric(data.calcium_discharge_value, errors='coerce')
# data.sodium_admit_value = pd.to_numeric(data.sodium_admit_value, errors='coerce')
# data.sodium_discharge_value = pd.to_numeric(data.sodium_discharge_value, errors='coerce')
# data.bun_admit_value = pd.to_numeric(data.bun_admit_value, errors='coerce')
# data.bun_discharge_value = pd.to_numeric(data.bun_discharge_value, errors='coerce')
# data.albumin_admit_value = pd.to_numeric(data.albumin_admit_value, errors='coerce')
# data.albumin_discharge_value = pd.to_numeric(data.albumin_discharge_value, errors='coerce')
# data.hemoglobin_admit_value = pd.to_numeric(data.hemoglobin_admit_value, errors='coerce')
# data.hemoglobin_discharge_value = pd.to_numeric(data.hemoglobin_discharge_value, errors='coerce')

# data = data[data.bmi_discharge < 205]
# data = data[data.bmi_admit < 205]
# # peak BMI on record was 204 https://en.wikipedia.org/wiki/List_of_the_heaviest_people
# data = data[data.calcium_admit_value < 40]
# data = data[data.calcium_discharge_value < 40]
# data = data[data.sodium_admit_value < 300]
# data = data[data.sodium_discharge_value < 300]
# data = data[data.bun_admit_value < 500]
# data = data[data.bun_discharge_value < 500]
# data = data[data.albumin_admit_value < 40]
# data = data[data.albumin_discharge_value < 40]
# data = data[data.hemoglobin_admit_value < 40]
# data = data[data.hemoglobin_discharge_value < 40]

# print(data["length_of_stay_in_days"].describe())

# print(list(data))

print("Saving to file...")
filename1 = config.CLEAN_PHASE_02
data.to_pickle(filename1)
print("Clean phase_02 available at:", filename1)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
