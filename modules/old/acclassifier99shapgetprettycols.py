import glob
import os
import traceback
from pathlib import Path

import h5py
import pandas as pd

from cbh import config

print("Loading path to files...")
model_dir = config.MODELS_DIR
shap_csv_dir = config.SHAP_CSV_DIR
prettified_csv_dir = shap_csv_dir / "prettified"
tmp_csv_dir = prettified_csv_dir / "tmp"
datestr = "2019-05-23"
dirpath = model_dir  # / datestr
df_list = []


def make_pretty_cols():
    for filename in Path(dirpath).glob("**/**/*.h5"):

        try:
            justname = os.path.split(filename)[1]
            savefile = Path(prettified_csv_dir / justname)
            f = h5py.File(Path(filename), "r")
            # keylist = list(f.keys())
            # print("This h5 file contains", keylist)
            shap_expected_value = pd.read_hdf(
                Path(f"{filename}"), key="shap_expected_value"
            )
            target = shap_expected_value.iloc[0]["target"]
            pretty_imp_cols = pd.read_hdf(Path(f"{filename}"), key="pretty_imp_cols")
            pretty_imp_cols.to_csv(
                f"{savefile}.csv",
                header=[f"{target} SHAP Values in Order of Descending Importance"],
            )
        except Exception as exc:
            print(traceback.format_exc())


def merge_pretty_cols():
    for filename in Path(tmp_csv_dir).glob("*.csv"):

        try:
            df_list.append(pd.read_csv(filename))
        except Exception as exc:
            print(traceback.format_exc())

    try:
        df = pd.concat(df_list, axis=1)
        # print(df_list)
        # print(df.head())
        df.to_csv("combotest.csv")
        print("Done.")
    except Exception as exc:
        print(traceback.format_exc())


# First:
# make_pretty_cols()

# Then run that beautiful SED command over in Arch land,
# delete the CSVs you don't want in the tmp folder, and finally:
merge_pretty_cols()
