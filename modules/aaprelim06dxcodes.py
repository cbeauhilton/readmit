import json
import os
import sys
import zipfile
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.request import urlopen
import requests

import censusdata
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from tqdm import tqdm
import csv

tqdm.pandas()

sys.path.append("modules")
from cbh import config
import configcols

print("About to run", os.path.basename(__file__))
startTime = datetime.now()


# display all columns when viewing dataframes: make the number
# anything bigger than your number of columns
pd.options.display.max_columns = 2000

'''
Download and merge ICD10 first, then ICD9
'''

if not os.path.isfile(config.ICD10_DATABASE):
    print("Downloading ICD10 code CSV from Github...")
    url = "https://raw.githubusercontent.com/kamillamagna/ICD-10-CSV/master/codes.csv"
    data = requests.get(url)
    with open(config.ICD10_DATABASE, "w") as f:
        writer = csv.writer(f)
        reader = csv.reader(data.text.splitlines())
        for row in reader:
            writer.writerow(row)
    # df0 = pd.read_csv(config.ICD10_DATABASE, header=None)
    # df0.to_csv(config.ICD10_DATABASE)
    print("CSV downloaded, available at", config.ICD10_DATABASE)
else:
    print("Loading ICD10 code CSV...")


filename = config.ICD10_DATABASE
df0 = pd.read_csv(filename, header=None)
print("Cleaning ICD10 code dataframe...")
df0 = df0.drop([1, 3, 5], axis=1)
df0 = df0.rename(index=str, columns={2: "diagnosis code", 4: "diagnosis description"})
# print("ICD10: ", len(df0))
# Some of the diagnosis codes in the CCF dataset are in column 0 of the ICD database,
# others are in column 2.
# So do two separate merges, then put it all together.
df0_0 = df0[[0, "diagnosis description"]].copy()
# df0_0 = pd.DataFrame(df0_0)
df0_0 = df0_0.rename(index=str, columns={0: "diagnosis code"})

df0 = df0.drop([0], axis=1)
# print(df0.head())
# print(df0_0.head())

ccf_code_list = Path(
    r"C:\Users\hiltonc\Desktop\readmit\readmit\docs\value counts\primary_diagnosis_code.csv"
)
df1 = pd.read_csv(ccf_code_list)
print("CCF:", len(df1))
df1 = df1.drop(["primary_diagnosis_code", "Unnamed: 0"], axis=1)
df1 = df1.rename(index=str, columns={"index": "diagnosis code"})
df1 = df1.sort_values(by=["diagnosis code"])
# print(df1.head())

# Merge CCF data and the first ICD dataframe
df = pd.merge(df0, df1, on="diagnosis code", indicator=True, how="outer")
# df = df.rename(index=str, columns={"_merge": "_merge0"})
# df = df.dropna(subset=["index"])
# df["code"] = df[4].astype(str)
df = df.drop(["_merge"], axis=1)
# print(df.tail(10)) # shows the missing dx descriptions

# Pull out the CCF data that didn't match, and merge with the second ICD dataframe
df2 = df[df["diagnosis description"].isnull()]
# print(len(df2))
df2 = df2.drop(["diagnosis description"], axis=1)
df3 = pd.merge(df2, df0_0, on="diagnosis code", indicator=True, how="outer")
df3 = df3.drop(["_merge"], axis=1)
# print("DF3:", df3.tail())
# print("DF3:", len(df3))

# Now put them all together
# print("DF:", len(df))
df4 = df[df["diagnosis description"].notnull()]
# print("DF4:", len(df4))
dffinal = pd.concat(
    [df4, df3], ignore_index=True, keys=["x", "y"], verify_integrity=True, axis=0
)
dffinal = dffinal.drop_duplicates(subset=["diagnosis code"], keep="first")
# print(dffinal.tail(100))
# print(len(dffinal))
# print(dffinal.count())


# fix values wrt casing, bad spacing, etc.
print("Cleaning text within cells...")
dffinal = dffinal.progress_apply(
    lambda x: x.str.strip()
    .str.replace("\t", "")
    .str.replace(", ", " ")
    .str.replace(",", " ")
    .str.replace("'", "")
    if (x.dtype == "object")
    else x
)

# dffinal.to_pickle(config.DX_CODES_CONVERTED)
# print("File saved to", config.DX_CODES_CONVERTED)


'''
ICD 9 codes
'''

filename = Path(r"C:\Users\hiltonc\Desktop\readmit\readmit\docs\DDW_Diagnoses.csv")
ddw = pd.read_csv(filename)
ddw = ddw.rename(index=str, columns={"Diagnosis_Code": "diagnosis code", "Diagnosis_Description": "diagnosis description"})
ddw = ddw.drop(["Diagnosis_Key"], axis=1)
# print(ddw.CodeSet.unique())
ddw = ddw[ddw["CodeSet"]=="9"]
ddw = ddw.drop(["CodeSet"], axis=1)



# grab rows with missing values - these are the ICD9 codes
dffinalnull = dffinal[dffinal["diagnosis description"].isnull()]
dffinalnull = dffinalnull.drop(["diagnosis description"], axis=1)

df99 = pd.merge(dffinalnull, ddw, on="diagnosis code", indicator=False, how="right")
df100 = pd.concat(
    [df99, dffinal], ignore_index=True, keys=["x", "y"], verify_integrity=True, axis=0
)
df001 = df100.sort_values(by=["diagnosis code"])
df100 = df100.drop_duplicates(subset=["diagnosis code"], keep="first")
print(len(df100))
print(df100.head())
df100.to_pickle(config.DX_CODES_CONVERTED)
print("File saved to", config.DX_CODES_CONVERTED)
# di = pd.Series(
#     dffinal["diagnosis description"].values, index=dffinal["diagnosis code"]
# ).to_dict()
# print(di)

# dxcodes = pd.read_pickle(config.DX_CODES_CONVERTED)
# print(len(dxcodes))


# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
print("\n")
