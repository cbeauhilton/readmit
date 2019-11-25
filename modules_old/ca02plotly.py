#%%

import os
import sys
from datetime import datetime

import cufflinks as cf
import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot

sys.path.append("modules")

import config

# Select offline option and set the global theme for cufflinks
cf.go_offline()
cf.set_config_file(world_readable=True, theme="pearl", offline=True)

# Always run this the command before at the start of notebook
init_notebook_mode(connected=True)

# print("About to run ", os.path.basename(__file__))
startTime = datetime.now()

# display all columns when viewing dataframes: make the number
# anything bigger than your number of columns
pd.options.display.max_columns = 2000

figures_path = config.FIGURES_DIR
tables_path = config.TABLES_DIR
filename = config.PROCESSED_FINAL_DESCRIPTIVE

#%%

print("Loading ", filename)
data = pd.read_pickle(filename)

#%%
data0 = data
print(len(data))
data0 = data0[data0.length_of_stay_in_days < 31]
print(len(data0))
data0 = data0[data0.length_of_stay_in_days > 0]
data0.length_of_stay_in_days = data0.length_of_stay_in_days.apply(np.floor) #round down so the hist looks nice
print(len(data0))
data0 = data0[data0.patient_age < 110]  # https://en.wikipedia.org/wiki/Oldest_people
print(len(data0))
data0 = data0[data0.patient_age >= 0]
print(len(data0))

data1 = data0
a = np.array(data1["days_between_current_discharge_and_next_admission"].values.tolist())
data1["days_between_current_discharge_and_next_admission"] = np.where(
    a > 31, np.nan, a
).tolist()
# data1.days_between_current_discharge_and_next_admission = data1.days_between_current_discharge_and_next_admission.apply(
#     np.floor
# ) # round down so the hist looks nice
print(len(data1))
data1 = data1[
    (data1.days_between_current_discharge_and_next_admission > 0.1)
    | (data1.days_between_current_discharge_and_next_admission == np.nan)
]
print(len(data1))


#%%
# Generate interactive plots

print("patient_age")
data0["patient_age"].iplot(
    kind="hist",
    xTitle="Age",
    yTitle="count",
    title="Age Distribution",
    filename=os.path.join(figures_path, "pt_age"),
    asPlot=True,
)

#%%
print("length_of_stay")
data0["length_of_stay_in_days"].iplot(
    kind="hist",
    bins=31,
    xTitle="Days",
    yTitle="Count",
    title="Length of Stay Distribution",
    filename=os.path.join(figures_path, "los"),
    asPlot=True,
)

#%%

print("days_between_current_discharge_and_next_admission")
data1["days_between_current_discharge_and_next_admission"].iplot(
    kind="hist",
    xTitle="Days",
    yTitle="Count",
    title="days_between_current_discharge_and_next_admission",
    filename=os.path.join(
        figures_path, "days_between_current_discharge_and_next_admission"
    ),
    asPlot=True,
)
#%%
print("admit_year")
data["admissiontime_year"].iplot(
    kind="hist",
    xTitle="Admit Year",
    yTitle="count",
    title="Admit Year Distribution",
    filename=os.path.join(figures_path, "admit_year"),
    asPlot=True,
)
#%%
data["this_is_hospital_encounter"].groupby(
    data["admissiontime"].dt.to_period("M")
).sum().iplot(
    filename=os.path.join(figures_path, "this_is_hospital_encounter"), asPlot=True
)

#%%

data["insurance2"].iplot(
    kind="hist",
    xTitle="Insurance",
    yTitle="CategoricalEncoder Count",
    title="Insurance Distribution",
    filename=os.path.join(figures_path, "insurance"),
    asPlot=True,
)

#%%

#%%
data.iplot(
    x="admissiontime_year",
    y="readmitted30d",
    # Specify the category
    categories="patientclassdescription",
    xTitle="Read Time",
    yTitle="Reading Percent",
    title="Reading Percent vs Read Ratio by Publication",
    filename=os.path.join(figures_path, "insurances"),
    asPlot=True,
)

#%%
df2 = (
    data["insurance2"].astype(str).value_counts()[:]
)  # plotly and cufflinks don't like the "category" dtype,
# so if you're using that dtype this will convert it to the proper "object" dtype

print("insurance2")
df2.iplot(
    kind="bar",
    yTitle="Number of Patients",
    title="Insurance",
    filename=os.path.join(figures_path, "insurance2"),
    asPlot=True,
)
#%%
# This makes a graph of the mean LoS
df2 = (
    data[["length_of_stay_in_days", "admissiontime"]]
    .set_index("admissiontime")
    .resample("M")
    .mean()
)

df2 = df2["2012-01-01":"2018-07-01"]
print("length_of_stay")
df2.iplot(
    mode="lines+markers+text",
    xTitle="Date",
    yTitle="Count",
    title="Length of Stay",
    filename=os.path.join(figures_path, "mean_LoS_by_month"),
    asPlot=True,
)

#%%
# print("length_of_stay")
# # This makes overlapping histograms showing Hgb at admit and discharge
# data[["hemoglobin_admit", "hemoglobin_discharge"]].iplot(
#     kind="hist",
#     # histnorm='percent',
#     barmode="overlay",
#     # xTitle='Calcium at Admit and Discharge',
#     # yTitle='(%) of Articles',
#     title="hemoglobin",
#     filename=os.path.join(figures_path, "hgb"),
#     asPlot=True,
# )


# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
