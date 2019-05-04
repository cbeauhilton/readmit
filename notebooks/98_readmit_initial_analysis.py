#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Table 1 and Exploratory Data Analysis

#%%
get_ipython().run_line_magic('load_ext', 'blackcellmagic')


#%%
import os
import sys
from collections import Counter
from itertools import count

import category_encoders as ce
import cufflinks
import hyperdash
import imblearn
import lightgbm as lgb
import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import requests
import seaborn as sns
import shap
import sklearn
from hyperdash import monitor_cell
from imblearn.metrics import classification_report_imbalanced
from IPython.core.display import HTML, display
from IPython.core.pylabtools import figsize
from lightgbm.sklearn import LGBMRegressor
from pandas import HDFStore
from plotly.offline import iplot
from sklearn import metrics
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import (
    CategoricalEncoder,
    Imputer,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.svm import SVR

# Select offline option and set the global theme for cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme="pearl", offline=True)

# Print versions
print("python: {}".format(sys.version))
print("pandas: {}".format(pd.__version__))
print("numpy: {}".format(np.__version__))
print("scikit learn: {}".format(sklearn.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("seaborn: {}".format(sns.__version__))
print("lightgbm: {}".format(lgb.__version__))
print("shap: {}".format(shap.__version__))

# prevent chained assignment error message
pd.options.mode.chained_assignment = None  # default='warn'

# print the JS visualization code to the notebook
shap.initjs()

pd.options.display.max_columns = 2000
pd.options.display.max_rows = 2000

# os.getcwd()


#%%
import config
filename = "C:\\Users\\hiltonc\\Desktop\\readmit\\data\\raw\\ccf_rawdata.h5"
filename = config.RAW_DATA_FILE
filename


#%%
store = HDFStore(filename)
store.keys()


#%%
data = store["ccf_raw"]  # load it
data


#%%
data = data.apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)


#%%
# make possibly interesting time variables
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
data["admit_day_of_week"] = data["admissiontime"].dt.weekday_name
data["discharge_day_of_week"] = data["dischargetime"].dt.weekday_name
data["admit_year"] = data["admissiontime"].dt.year
cal = calendar()
holidays = cal.holidays(start="2000-01-01", end="2050-12-31")
data["admitted_on_holiday"] = data["admissiontime"].isin(holidays)
data["discharged_on_holiday"] = data["dischargetime"].isin(holidays)


#%%
data.update(data[["ed_admission",
        "diff_type_discharge_value",
        "lineinfection",
        "cdiffinfection",
        "fallduringadmission",
        "ondialysis",
        "opiatesduringadmit",
        "benzosduringadmit",
        "pressureulcer",
        "dischargedonbenzo",
        "dischargedonopiate",
        "myocardial_infarction",
        "congestive_heart_failure",
        "peripheral_vascular_disease",
        "cerebrovascular_disease",
        "dementia",
        "chronic_pulmonary_disease",
        "rheumatic_disease",
        "peptic_ulcer_disease",
        "mild_liver_disease",
        "diabetes_without_chronic_complication",
        "diabetes_with_chronic_complication",
        "hemiplegia_or_paraplegia",
        "renal_disease",
        "any_malignancy,including_lymphoma_and_leukemia,except_malignant_neoplasm_of_skin",
        "moderate_or_severe_liver_disease",
        "metastatic_solid_tumor",
        "aids/hiv",
        "connective_tissue_disorder",
        "pneumonia",
        "depression",
        "anxiety",
        "psychosis",
        "cerebral_palsy",
        "short_gut_syndrome",
        "epilepsy",
        "knee_replacement",
        "hip_replacement",
        "solid_organ_transplant",
        "tpn",
        "pt_ot_consult",
        "spiritualcareconsult",
        "palliativecareconsult",
        "infectiousdiseaseconsult",
        "hospiceconsult",
                 ]].fillna(0))

data


#%%
data = data.rename_axis('patientid')


#%%
print("Total number of unique admissions:", len(data))
print("Total number of unique patients:", data.index.nunique())


#%%
# new data frame with split value columns 
new = data["admitbp"].str.split("/", n = 1, expand = True) 
  
# making seperate first name column from new data frame 
data["admit_systolic_bp"]= new[0] 
  
# making seperate last name column from new data frame 
data["admit_diastolic_bp"]= new[1] 
  
# Dropping old Name columns 
data.drop(columns =["admitbp"], inplace = True) 

# df display 
data 

# data["admitbp"].str.rsplit("/", expand = True)


#%%
# new data frame with split value columns 
new = data["dischargebp"].str.split("/", n = 1, expand = True) 
  
# making seperate first name column from new data frame 
data["discharge_systolic_bp"]= new[0] 
  
# making seperate last name column from new data frame 
data["discharge_diastolic_bp"]= new[1] 
  
# Dropping old Name columns 
data.drop(columns =["dischargebp"], inplace = True) 


#%%
data


#%%
print("Fixing dtypes...")
data_bool = data.select_dtypes(["bool"])
converted_bool = data_bool * 1.0  # changes bool to int

data_int = data.select_dtypes(include=["int"])
converted_int = data_int.apply(pd.to_numeric, downcast="unsigned")

data_float = data.select_dtypes(include=["float"])
converted_float = data_float.apply(pd.to_numeric, downcast="float")

data[converted_int.columns] = converted_int
data[converted_float.columns] = converted_float
data[converted_bool.columns] = converted_bool


# data_obj = data.select_dtypes(include=["object"]).copy()
# converted_obj = pd.DataFrame()
# for col in data_obj.columns:
#     num_unique_values = len(data_obj[col].unique())
#     num_total_values = len(data_obj[col])
#     if num_unique_values / num_total_values < 0.5:
#         converted_obj.loc[:, col] = data_obj[col].astype("category")
#     else:
#         converted_obj.loc[:, col] = data_obj[col]

# data[converted_obj.columns] = converted_obj


#%%
data.fillna(value=pd.np.nan, inplace=True)
data.replace(to_replace=[None], value=np.nan, inplace=True)
data.dtypes


#%%
data


#%%
np.set_printoptions(threshold=np.nan)
g = data.columns.to_series().groupby(data.dtypes).groups
{k.name: v for k, v in g.items()}


#%%
data = data['2011-01-01':'2018-12-31']


#%%
data.describe()


#%%
#data.loc[:, 'calcium_admit_value':'albumin_discharge_value'].hist(bins=100)

#data.loc[:, 'calcium_admit_value':'albumin_discharge_value'].iplot(kind='histogram', subplots=True, #shape=(10, 1)
#                                                                  )


#%%
# % of pts w a given condition (mean works bc: binary (1 or 0) over whole length of data)

data.loc[:, "myocardial_infarction":"solid_organ_transplant"].mean()


#%%
# print(data['patient_age'].value_counts())


#%%
# Remove extreme outliers
data = data[data.bmi < 205]
# peak BMI on record was 204 https://en.wikipedia.org/wiki/List_of_the_heaviest_people
data = data[data.patient_age < 130]  # https://en.wikipedia.org/wiki/Oldest_people
data = data[data.patient_age > 0]  # some negative ages snuck in somehow
data = data[data.calcium_admit_value < 40]
data = data[data.calcium_discharge_value < 40]
data = data[data.sodium_admit_value < 300]
data = data[data.sodium_discharge_value < 300]
data = data[data.bun_admit_value < 500]
data = data[data.bun_discharge_value < 500]
data = data[data.albumin_admit_value < 40]
data = data[data.albumin_discharge_value < 40]
data = data[data.hemoglobin_admit_value < 40]
data = data[data.hemoglobin_discharge_value < 40]

# data = data[data.bmi.quantile(.99)]
# print(data['bmi'].describe())
# print(data['bmi'].value_counts())


#%%
data[["calcium_admit_value", "calcium_discharge_value"]].iplot(
    kind="hist",
    # histnorm='percent',
    barmode="overlay",
    # xTitle='Calcium at Admit and Discharge',
    # yTitle='(%) of Articles',
    title="calcium",
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\calcium",
    asPlot=True,
)  # asPlot saves to HTML file, keeps interactivity, does not show inline


#%%
data[["sodium_admit_value", "sodium_discharge_value"]].iplot(
    kind="hist",
    # histnorm='percent',
    barmode="overlay",
    # xTitle='Calcium at Admit and Discharge',
    # yTitle='(%) of Articles',
    title="sodium",
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\sodium",
    asPlot=True,
)


#%%
data[['bun_admit_value', 'bun_discharge_value']].iplot(
    kind='hist',
    #histnorm='percent',
    barmode='overlay',
    #xTitle='Calcium at Admit and Discharge',
    #yTitle='(%) of Articles',
    title='bun',
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\bun",
    asPlot=True)


#%%
data[['albumin_admit_value', 'albumin_discharge_value']].iplot(
    kind='hist',
    #histnorm='percent',
    barmode='overlay',
    #xTitle='Calcium at Admit and Discharge',
    #yTitle='(%) of Articles',
    title='albumin',
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\albumin",
    asPlot=True)


#%%
data[['hemoglobin_admit_value', 'hemoglobin_discharge_value']].iplot(
    kind='hist',
    #histnorm='percent',
    barmode='overlay',
    #xTitle='Calcium at Admit and Discharge',
    #yTitle='(%) of Articles',
    title='hemoglobin',
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\hemoglobin",
    asPlot=True)


#%%
data


#%%
data['patient_age'].iplot(kind='hist', xTitle='Age',
                  yTitle='count', title='Age Distribution',
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\age",
    asPlot=True
                          )


#%%
data['bmi'].iplot(kind='hist', xTitle='BMI',
                  yTitle='count', title='BMI Distribution',
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\bmi",
    asPlot=True
                          )


#%%
data['sbp'].iplot(kind='hist', xTitle='SBP',
                  yTitle='count', title='Systolic Blood Pressure Distribution',
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\SBP",
    asPlot=True
                          )


#%%
data['dbp'].iplot(kind='hist', xTitle='DBP',
                  yTitle='count', title='Diastolic Blood Pressure Distribution',
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\DBP",
    asPlot=True
                          )


#%%
data['length_of_stay'].iplot(kind='hist', xTitle='Length of Stay (Days)',
                  yTitle='count', title='Length of Stay Distribution',
    filename="C:\\Users\\hiltonc\\Desktop\\readmit\\reports\\figures\\LoS",
    asPlot=True
                          )


#%%
ax = pd.value_counts(data['race']).plot(kind='bar', 
                                         figsize=(7,6), 
                                         ylim=(0,1_100_000), 
                                         x="Race", 
                                         y="Count", 
                                         title="Racial Distribution"
                                        )
for p in ax.patches: ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')


#%%
ax = pd.value_counts(data["primary_language"]).plot(
    kind="bar",
    figsize=(70, 6),
    ylim=(0, 1_800_000),
    x="Race",
    y="Count",
    title="Primary Language",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%
ax = pd.value_counts(data["marital_status"]).plot(
    kind="bar",
    figsize=(7, 6),
    ylim=(0, 1_100_000),
    x="Race",
    y="Count",
    title="Marital Status",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%


print(data["insurance2"].nunique())
ax = pd.value_counts(data["insurance2"]).plot(
    kind="bar",
    figsize=(300, 6),
    ylim=(0, 150_000),
    x="Race",
    y="Count",
    title="insurance2",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%
ax = pd.value_counts(data["attending_specialty_institute_desc"]).plot(
    kind="bar",
    figsize=(20, 6),
    ylim=(0, 800_000),
    x="Race",
    y="Count",
    title="attending_specialty_institute_desc",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%
ax = pd.value_counts(data["patientclassdescription"]).plot(
    kind="bar",
    figsize=(7, 6),
    ylim=(0, 1_200_000),
    x="Race",
    y="Count",
    title="patientclassdescription",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%
ax = pd.value_counts(data["dischargedispositiondescription"]).plot(
    kind="bar",
    figsize=(15, 6),
    ylim=(0, 900_000),
    x="Race",
    y="Count",
    title="dischargedispositiondescription",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%
print(data["primary__diagnosis_code"].nunique())
print(data["primary__diagnosis_desc"].nunique())
# way too many diagnoses to plot
# ax = pd.value_counts(data['primary__diagnosis_code']).plot(kind='bar', figsize=(150,6), ylim=(0,1100000), x="Race", y="Count", title="primary__diagnosis_code")
# for p in ax.patches: ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')


#%%
data


#%%
yearly = (
    data.set_index("admit_date").groupby(pd.Grouper(freq="Y"))["readmittedany"].count()
)
ax = yearly.plot(
    kind="bar",
    figsize=(6, 6),
    ylim=(0, 200_000),
    x="Year",
    y="Number of Hospitalizations",
    title="Hospitalization Counts Over Time",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%
yearly = (
    data.set_index("admit_date").groupby(pd.Grouper(freq="Y"))["readmittedany"].sum()
)
ax = yearly.plot(
    kind="bar",
    figsize=(6, 6),
    ylim=(0, 50000),
    x="Year",
    y="Number of Readmissions",
    title="Readmission Counts Over Time",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%
yearly = (
    data.set_index("admit_date").groupby(pd.Grouper(freq="Y"))["readmittedany"].mean()
)
ax = yearly.plot(
    kind="bar",
    figsize=(6, 6),
    ylim=(0, 0.5),
    x="Year",
    y="Number of Readmissions",
    title="Readmission Average Over Time",
)
for p in ax.patches:
    ax.annotate(
        np.round(p.get_height(), decimals=2),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )


#%%
yearlyMeanLoS = (
    data.set_index("admit_date").groupby(pd.Grouper(freq="M"))["length_of_stay"].mean()
)
ax = yearlyMeanLoS.plot(
    kind="line", figsize=(5, 5), ylim=(0, 10), title="Mean Length of Stay Over Time"
)
# for p in ax.patches: ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')


#%%
data["year"] = data.admit_date.dt.year
games_per_day = data.pivot_table(
    index="year", columns="admit_day_of_week", values="admit_date", aggfunc=len
)
games_per_day = games_per_day.divide(games_per_day.sum(axis=1), axis=0)

ax = games_per_day.plot(kind="area", stacked="true")
ax.legend(loc="upper right")
ax.set_ylim(0, 1)
plt.show()

#%% [markdown]
# # Alex on proportion of admissions accounted for
# 9% of relevant admissions are excluded because they don’t exist in billing (relevant = inpatient or obs, not Florida, not ACMC, not newborns, not bad preadmission testing visits)
#                 Mostly babies & non-main/Hospital/FHC locations
#                 or older admissions at non-main/HC/FV/LK hospitals (aka Medina)
#                 8% medina
#                 10% OSH surgeries
#                 12% Hillcrest L&D
# 

