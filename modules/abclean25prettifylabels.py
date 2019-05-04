import os
import sys
import glob
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
sys.path.append("modules")
import config
import configcols


shap_names = [
    "financialclass_binary_shap",
    "gender_binary_shap",
    "race_binary_shap",
    "length_of_stay_over_3_days_shap",
    "length_of_stay_over_5_days_shap",
    "length_of_stay_over_7_days_shap",
    "length_of_stay_over_14_days_shap",
    "readmitted3d_shap",
    "readmitted7d_shap",
    "readmitted30d_shap",
]

rootpath = config.PROCESSED_DATA_DIR

list_of_shaps = glob.glob(f'{config.PROCESSED_DATA_DIR}/**/**/*_shap.csv', recursive=True)

df = pd.concat((pd.read_csv(f) for f in list_of_shaps))
df = df.rename(index=str, columns={"0": "feature"})
df = df.drop(["Unnamed: 0"], axis=1)
df = df.drop_duplicates(keep="first")


# fix values wrt casing, bad spacing, etc.
print("Cleaning text within cells...")
df = df.progress_apply(
    lambda x: x.str.strip()
    .str.replace("\t", "")
    .str.replace("_", " ")
    .str.replace("__", " ")
    .str.replace(", ", " ")
    .str.replace(",", " ")
    .str.replace("'", "")
    .str.capitalize()
    if (x.dtype == "object")
    else x
)

print(len(df))
print(df.head(50))

csv_name = rootpath / "prettifying.csv"
df.to_csv(csv_name)

# dxcodes = pd.read_pickle(dxcode_file)

# di = pd.Series(
#     dxcodes["diagnosis description"].values, index=dxcodes["diagnosis code"]
# ).to_dict()
# result["primary_diagnosis_code"].map(di).fillna(
#     result["primary_diagnosis_code"]
# )