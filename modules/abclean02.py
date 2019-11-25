import sys

sys.path.append("modules")

from datetime import datetime

import os
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

tqdm.pandas()

from cbh import config
from cbh import configcols

print("About to run", os.path.basename(__file__))
startTime = datetime.now()

pd.options.display.max_columns = 2000

# Load file
# filename = config.CLEAN_PHASE_01
# print("Loading", filename)
# data = pd.read_pickle(filename)
# print("File loaded.")

interim_file = config.INTERIM_H5
print("Loading", interim_file)
data = pd.read_hdf(interim_file, key='phase_01')
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

# fix the sign on this column
data["length_of_stay_of_last_admission"] = (
    data["length_of_stay_of_last_admission"] * -1
)  

# old LoS column no longer needed, the new one is better
data = data.drop(["length_of_stay"], axis=1)  

print("Saving to file...")
filename1 = config.CLEAN_PHASE_02
data.to_pickle(filename1)
print("Clean phase_02 available at:", filename1)
data.to_hdf(interim_file, key='phase_02', mode='a', format='table')

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
