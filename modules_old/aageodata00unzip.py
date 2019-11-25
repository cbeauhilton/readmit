import json
import os
import sys
import zipfile
from datetime import datetime
from io import StringIO
from urllib.request import urlopen

import pandas as pd
from pandas.io.json import json_normalize
from tqdm import tqdm

sys.path.append("modules")
from cbh import config

tqdm.pandas()

print("About to run ", os.path.basename(__file__))
startTime = datetime.now()


archive = config.GEODATA_RAW_ZIP
interim_dir = config.INTERIM_DATA_DIR

print("Extracting zip file...")
zip_ref = zipfile.ZipFile(archive, "r")
extracted = zip_ref.namelist()
zip_ref.extractall(interim_dir)
zip_ref.close()
extracted_file = os.path.join(interim_dir, extracted[0])
print(f"Zip file extracted to {extracted_file}")

print("Reading text file into dataframe...")
geo_raw = pd.read_csv(extracted_file, sep="\t", header=0, lineterminator='\r')

print("Dataframe created.")

print("Setting datetime columns to correct dtypes...")
geo_raw = geo_raw.progress_apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)

print("Saving NaNs as Not Available...")
geo_raw = geo_raw.fillna(value="Not available") # missingness can be informative, so keep it
geo_raw["GEOID"] = geo_raw["GEOID"].astype(str) # the others get read in as strings just fine

print(geo_raw.dtypes)

print("...done.")

print("Saving to pickle...")
file_name = config.GEODATA_RAW
geo_raw.to_pickle(file_name)
print("Pickle file available at", file_name)

# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
