#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os

try:
    os.chdir(os.path.join(os.getcwd(), "notebooks"))
    print(os.getcwd())
except:
    pass

#%%
# Imports
import sys

sys.path.append("C:\\Users\\hiltonc\\Desktop\\readmit\\modules")

import config
import importlib

importlib.reload(config)
import numpy as np
import pandas as pd
from pandas import HDFStore
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()

pd.options.display.max_columns = 2000
pd.options.display.max_rows = 2000

# fix random seed for reproducibility
seed = config.SEED

np.random.seed(seed)

#%%
# file_name = config.CLEAN_PHASE_03
# file_name = config.CLEAN_PHASE_04
# file_name = config.GEODATA_BLOCK_INFO
# file_name = config.GEODATA_FINAL
# file_name = config.RAW_DATA_FILE
file_name = config.CLEAN_PHASE_09
# file_name = config.DX_CODES_CONVERTED
data = pd.read_pickle(file_name)
#%%
# print(data.count())
# data = data[data["primary_diagnosis_code"] == "6826"]
# data = data[data["ACS_BlockGroup_geoid"] == "15000US390351246004"]
# print(data.dtypes)
# print(len(data))
# print(data.financialclass.nunique())
# data


# feature_list = list(data)
# df = pd.DataFrame(feature_list, columns=["features"])
# df.to_csv("features.csv", index=False)

#%%
# data0 = data[
#     ["acs_total_population_count","acs_race_white_alone","acs_200_and_over_ratio_income_poverty_level_past_12_mo"
#     ]
# ]
# data0 = data0.astype("float")
# data0["over_200_ratio"] = data0["acs_200_and_over_ratio_income_poverty_level_past_12_mo"]/data0["acs_total_population_count"]
# data0["white_alone_ratio"] = data0["acs_race_white_alone"]/data0["acs_total_population_count"]
# data1 = data #[data0["icu_length_of_stay_in_days"] > data0["length_of_stay_in_days"]]
# data1["icuminuslos"] = data1["icu_length_of_stay_in_days"] - data1["length_of_stay_in_days"]
# data1 = data1[data1["days_between_current_discharge_and_next_admission"] > 0]
# data1 = data1[data1["days_between_current_discharge_and_next_admission"] < 90]
# data = data[data["length_of_stay_in_days"] < 0]
# data = data.sort_index(axis=1)
# data = data.sort_values(["patientid", "admissiontime"])
# data.heartrate_admit.value_counts(sort=True, dropna=False, ).to_frame().reset_index().to_csv("column.csv")
# data1 = data1.sort_values(by=['icuminuslos'])
# a = np.array(data1["icuminuslos"].values.tolist())
# data1["icuminuslos"] = np.where(a > 2, np.nan, a).tolist()
# print(len(data1))
# data1.head(20)

#%%
data1 = data[["days_between_current_discharge_and_next_admission"]].copy()
a = np.array(data1["days_between_current_discharge_and_next_admission"].values.tolist())
data1["days_between_current_discharge_and_next_admission"]= np.where(a <=0, np.nan, a).tolist()
a = np.array(data1["days_between_current_discharge_and_next_admission"].values.tolist())
data1["days_between_current_discharge_and_next_admission"]= np.where(a >=1, np.nan, a).tolist()
data1 = data1.dropna()
data1.sort_values(by=["days_between_current_discharge_and_next_admission"])
# data1["days_between_current_discharge_and_next_admission"] = data1["days_between_current_discharge_and_next_admission"] <= 2

# data1.days_between_current_discharge_and_next_admission.describe()
# print(data1.days_between_current_discharge_and_next_admission.value_counts(dropna=False,normalize=True,bins=None,ascending=True))

# print(data1.days_between_current_discharge_and_next_admission.value_counts(dropna=False,normalize=True,bins=10,ascending=True))

print(data1.days_between_current_discharge_and_next_admission.value_counts(dropna=False,normalize=False,bins=10,ascending=True))
#%%
data1.days_between_current_discharge_and_next_admission.plot(kind="hist")
data1
# for column in data.columns:
#     data[column].value_counts(sort=True, dropna=False, ).to_frame().reset_index().to_csv(f"{column}.csv")
# result2 = data.progress_apply(pd.value_counts)
# result2

#%%

# tables_path = config.TABLES_ONE_DIR
# third_file = os.path.join(tables_path, "finaltable1.csv")
# fourth_file = os.path.join(tables_path, "finaltable.pickle")
# third = pd.read_csv(third_file)
# fourth = pd.read_pickle(fourth_file)

# #%%
# third.to_latex("final1.tex")
# third


# #%%
# tables_path = config.TABLES_ONE_DIR
# fourth_file = os.path.join(tables_path, "finaltable.pickle")
# fourth = pd.read_pickle(fourth_file)
# fourth

# #%%
# feature_selection_csv = Path(r"C:\Users\hiltonc\Desktop\readmit\reports\tables\featureselectiontrainingreports.csv")
# import config

# figures_path = config.FIGURES_DIR
# feat_sel = pd.read_csv(feature_selection_csv)
# print(feat_sel.columns)
# import cufflinks as cf
# from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
# cf.set_config_file(world_readable=True, theme="pearl", offline=True)
# init_notebook_mode(connected=True)
# cf.go_offline()

# feat_sel.iplot(subplots=True, subplot_titles=True)
# #%%
# feat_sel.iplot(
#     kind="bubble",
#     mode="markers",
#     x="Number of Features",
#     y="ROC AUC",
#     size="Brier Score Loss",
#     text="Target",
#     xTitle="Number of Features",
#     yTitle="ROC AUC",
#     filename=os.path.join(
#         figures_path, "Training Metrics wrt Number of Features"
#     ),
#     # asPlot=True,
# )