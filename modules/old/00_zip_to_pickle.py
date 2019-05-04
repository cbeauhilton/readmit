import os
import sys
import zipfile
from io import StringIO
from tqdm import tqdm
tqdm.pandas()

import pandas as pd

sys.path.append('modules')
import config

archive = config.RAW_ZIP_FILE

print("Extracting zip file...")
zip_ref = zipfile.ZipFile(archive, 'r')
extracted = zip_ref.namelist()
zip_ref.extractall(config.RAW_DATA_DIR)
zip_ref.close()
extracted_file = os.path.join(config.RAW_DATA_DIR, extracted[0])
print("Zip file extracted.")

# import mmap

# def get_num_lines(extracted_file):
#     fp = open(extracted_file, "r+")
#     buf = mmap.mmap(fp.fileno(), 0)
#     lines = 0
#     while buf.readline():
#         lines += 1
#     return lines


# file = config.RAW_TXT_FILE
print('Reading text file into dataframe...')
with open(extracted_file) as f:
    #for line in tqdm(f, total=get_num_lines(extracted_file)):
            input = StringIO(f.read()
                        .replace('", ""', '')
                        .replace('"', '')
                        .replace(', ', ',')
                        .replace('\0', '')
                                )
    
            ccf_raw = pd.read_table(input, 
                                sep='\t', 
                                index_col=0,
                                engine='python'
                                )
      
print("Dataframe created.")

print('Setting datetime and timedelta columns to correct dtypes...')

ccf_raw = ccf_raw.progress_apply(
    lambda col: pd.to_datetime(col, errors="ignore") if (col.dtypes == object) else col,
    axis=0,
)

ccf_raw = ccf_raw.progress_apply(
    lambda col: pd.to_timedelta(col, errors="ignore")
    if (col.dtypes == object)
    else col,
    axis=0,
)

print('Basic text cleaning for columns...')
ccf_raw.columns = ccf_raw.columns.str.strip().str.lower().str.replace('  ', '_').str.replace(' ', '_').str.replace('__', '_')

print('clean')

print('Saving to pickle...')
file_name = config.RAW_DATA_FILE
ccf_raw.to_pickle(file_name)
print('Pickle file available at', file_name)