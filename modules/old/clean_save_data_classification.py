# coding: utf-8

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from io import StringIO
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas import HDFStore
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from helper_functions import StringIndexer, TypeSelector

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

# fix random seed for reproducibility
seed = config.SEED
np.random.seed(seed)

# Convert .txt file to .h5 file, after reading into Pandas Dataframe

# see https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data
# this .txt file needs a little finessing to load properly

file = config.RAW_TXT_FILE

with open(file) as f:
    input = StringIO(f.read()
                     .replace('", ""', '')
                     .replace('"', '')
                     .replace(', ', ',')
                     .replace('\0', '')
                               )
    
    ccf_raw = pd.read_table(input, 
                         sep='\t', 
                         index_col=0, 
                         header=None, 
                         engine='python'
                         )
      
print("open")

# basic text cleaning
ccf_raw.columns = ccf_raw.columns.str.strip().str.lower().str.replace('  ', '_').str.replace(' ', '_').str.replace('__', '_')
print('clean')

ccf_raw["admit_date"] = ccf_raw["admissiontime"].dt.date

# save to h5 file
store0 = HDFStore(config.RAW_DATA_FILE) # create store logic
store0['ccf_raw'] = ccf_raw  # save it
ccf_raw = store0['ccf_raw']  # load it
print("saved")

print("Loading file...")
store0 = HDFStore(config.RAW_DATA_FILE)  # create store logic
rawdata = store0["rawdata"]  # load it

# print(rawdata.info())

print("Cleaning...")
# Drop columns with large amount of missing data (.1 means 90% missing)
thresh = len(rawdata) * 0.01
rawdata = rawdata.dropna(thresh=thresh, axis=1, inplace=False)

# Fix all column names
rawdata.columns = (
    rawdata.columns.str.strip().str.lower().str.replace("  ", "_").str.replace(" ", "_")
)

# Automatically convert date columns to dates, leave the other ones alone
rawdata = rawdata.apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)

# fix values wrt casing, bad spacing, etc.
rawdata = rawdata.apply(
    lambda x: x.str.lower()
    .str.replace("\t", "")
    .str.replace("  ", "")
    .str.replace(" ", "")
    if (x.dtype == "object")
    else x
)

# Set index by date
rawdata = rawdata.reset_index().set_index(rawdata["admit_date"])
rawdata = rawdata.sort_index()


# (the "gender unknown" could become an "else" clause
# and don't convert to NaN until looking at the options)
gen = {"female": 1, "male": 0, "gender unknown": np.nan}
rawdata["gender"] = rawdata["gender"].replace(gen)

# make possibly interesting time variables
rawdata["admit_day_of_week"] = rawdata["admit_date"].dt.weekday_name
rawdata["discharge_day_of_week"] = rawdata["discharge_date"].dt.weekday_name
rawdata["admit_year"] = rawdata["admit_date"].dt.year
cal = calendar()
holidays = cal.holidays(start="2000-01-01", end="2050-12-31")
rawdata["admitted_on_holiday"] = rawdata["admit_date"].isin(holidays)
rawdata["discharged_on_holiday"] = rawdata["discharge_date"].isin(holidays)

# convert yes and no (false and true) to 1 and 0
d = {"y": 1, "n": 0, "yes": 1, "no": 0, "False": 0, "True": 1}
rawdata = rawdata.apply(lambda x: x.replace(d) if (x.dtype == "object") else x)

# print(rawdata.info())

# fix dtypes to save disk space, convert objects to categories if appropriate
print("Fixing dtypes...")
rawdata_bool = rawdata.select_dtypes(["bool"])
converted_bool = rawdata_bool * 1.0  # changes bool to int

rawdata_int = rawdata.select_dtypes(include=["int"])
converted_int = rawdata_int.apply(pd.to_numeric, downcast="unsigned")

rawdata_float = rawdata.select_dtypes(include=["float"])
converted_float = rawdata_float.apply(pd.to_numeric, downcast="float")

rawdata[converted_int.columns] = converted_int
rawdata[converted_float.columns] = converted_float
rawdata[converted_bool.columns] = converted_bool

rawdata_obj = rawdata.select_dtypes(include=["object"]).copy()
converted_obj = pd.DataFrame()
for col in rawdata_obj.columns:
    num_unique_values = len(rawdata_obj[col].unique())
    num_total_values = len(rawdata_obj[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:, col] = rawdata_obj[col].astype("category")
    else:
        converted_obj.loc[:, col] = rawdata_obj[col]

rawdata[converted_obj.columns] = converted_obj

print("Making PyTorch data...")
pytorch = rawdata
LABEL_COLUMN = config.LABEL_COLUMN

pytorch_train = pytorch[config.TRAIN_START : config.TRAIN_END]
pytorch_train_y = pytorch_train[LABEL_COLUMN]
pytorch_train_x = pytorch_train.drop([LABEL_COLUMN], axis=1)

pytorch_test = pytorch[config.TEST_START : config.TEST_END]
pytorch_test_y = pytorch_test[LABEL_COLUMN]
pytorch_test_x = pytorch_test.drop([LABEL_COLUMN], axis=1)

pytorch_valid = pytorch[config.VALID_START : config.VALID_END]
pytorch_valid_y = pytorch_valid[LABEL_COLUMN]
pytorch_valid_x = pytorch_valid.drop([LABEL_COLUMN], axis=1)

transformer = Pipeline(
    [
        (
            "features",
            FeatureUnion(
                n_jobs=1,
                transformer_list=[
                    # Part 1
                    (
                        "boolean",
                        Pipeline([("selector", TypeSelector("bool"))]),
                    ),  # booleans close
                    (
                        "numericals",
                        Pipeline(
                            [
                                ("selector", TypeSelector(np.number)),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                    ),  # numericals close
                    # Part 2
                    (
                        "categoricals",
                        Pipeline(
                            [
                                ("selector", TypeSelector("category")),
                                ("labeler", StringIndexer()),
                                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                    ),  # categoricals close
                ],
            ),
        )  # features close
    ]
)  # pipeline close

transformer.fit_transform(pytorch_train_x)
transformer.fit_transform(pytorch_test_x)

pytorch_train = pytorch_train_x
pytorch_train[LABEL_COLUMN] = pytorch_train_y

pytorch_test = pytorch_test_x
pytorch_test[LABEL_COLUMN] = pytorch_test_y

# Save for PyTorch
filename = config.PROCESSED_H5_PYTORCH
pytorch_train.to_hdf(filename, key="train", format="table")
pytorch_test.to_hdf(filename, key="test", format="table")
pytorch_valid.to_hdf(filename, key="valid", format="table")
print(
    "Pytorch data available at",
    config.PROCESSED_H5_PYTORCH,
    "via keys train, test, valid.",
)

print("Building train and test...")

# # One hot encoding
# rawdata = pd.get_dummies(rawdata, dummy_na=True, columns=None)

# set X and y for classification
c_labels = rawdata["isreadmittedasunplanned"]
c_features_no_dates = rawdata.drop(
    rawdata.select_dtypes(["datetime"]), inplace=False, axis=1
)
c_features = c_features_no_dates.drop(
    [
        "isreadmittedasunplanned",
        "isdeceased",
        # 'patientid',
    ],
    axis=1,
)

c_X = c_features
c_y = c_labels

# define training and test sets for classification
c_train_features, c_test_features, c_train_labels, c_test_labels = train_test_split(
    c_features, c_labels, stratify=c_labels, test_size=0.2, random_state=seed
)


# imputer = SimpleImputer(strategy="median")
# c_train_features = imputer.fit_transform(c_train_features)
#
# sampler = SMOTE(random_state=seed, kind="svm")
# c_train_features, c_train_labels = sampler.fit_sample(c_train_features,
# c_train_labels)
#
# c_train_features = pd.DataFrame(c_train_features)
# c_train_labels = pd.DataFrame(c_train_labels)
# c_test_features = pd.DataFrame(c_test_features)
# c_test_labels = pd.DataFrame(c_test_labels)
#
# print("imputed and upsampled")

# save to .h5 file in the "table" format,
# which is basically identical to Pandas DataFrame

print("Saving to .h5 file...")
filename = config.PROCESSED_H5_GBM

c_features.to_hdf(filename, key="c_features", format="table")
c_train_features.to_hdf(filename, key="c_train_features", format="table")
c_train_labels.to_hdf(filename, key="c_train_labels", format="table")
c_test_features.to_hdf(filename, key="c_test_features", format="table")
c_test_labels.to_hdf(filename, key="c_test_labels", format="table")
print("...complete.")
print("File available at", config.PROCESSED_H5_GBM)

c_train_features = pd.read_hdf(filename, key="c_train_features")
c_train_labels = pd.read_hdf(filename, key="c_train_labels")
c_test_features = pd.read_hdf(filename, key="c_test_features")
c_test_labels = pd.read_hdf(filename, key="c_test_labels")

# create lgb dataset files
print("Saving to LightGBM binary files...")
c_d_train = lgb.Dataset(c_train_features, label=c_train_labels)
c_d_test = lgb.Dataset(c_test_features, label=c_test_labels, reference=c_d_train)

c_X_train = c_train_features
c_y_train = c_train_labels
c_X_test = c_test_features
c_y_test = c_test_labels

try:
    os.remove(config.LIGHTGBM_TRAIN)
except OSError:
    pass

try:
    os.remove(config.LIGHTGBM_TEST)
except OSError:
    pass

c_d_train.save_binary(config.LIGHTGBM_TRAIN)
c_d_test.save_binary(config.LIGHTGBM_TEST)
print("...complete.")
print("Files available at", config.LIGHTGBM_TRAIN, "and", config.LIGHTGBM_TEST)
