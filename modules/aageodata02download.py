import glob
import json
import os
import sys
import tarfile
import time
import zipfile
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import censusdata
import pandas as pd
import requests
from pandas.io.json import json_normalize
from tqdm import tqdm

from cbh import config

sys.path.append("modules")

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.precision", 2)

tqdm.pandas()


print("About to run", os.path.basename(__file__))
startTime = datetime.now()


raw_file = config.GEODATA_RAW
interim_dir = config.INTERIM_DATA_DIR

# 150|00|US|39|035|1164|003
# summary_level| geographic component | country | state# | county | census tract| block group

# could get the years from the data directly,
# but 2018 isn't available and I don't want 2011
years = [2012,2013,2014,2015,2016,2017]

def download_census_geo_file(years):
    for year in years:
        geo_url = f'https://www2.census.gov/programs-surveys/acs/summary_file/{year}/data/5_year_entire_sf/{year}_ACS_Geography_Files.zip'
        # census.gov doesn't accept headerless requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36"}
        with requests.get(geo_url, stream=True, headers = headers) as r:
            local_filename = geo_url.split('/')[-1]
            local_filename = config.EXTERNAL_DATA_DIR/"census"/local_filename
            print(local_filename)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f:
                for chunk in (tqdm(r.iter_content(chunk_size=8192), total=total_size,unit='B', unit_scale=True)):  
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        # f.flush()
        return local_filename    

def download_census_blkgrp_file(years):
    for year in years:
        blkgrp_url = f'https://www2.census.gov/programs-surveys/acs/summary_file/{year}/data/5_year_entire_sf/Tracts_Block_Groups_Only.tar.gz'
        if year == 2017:
            # "tar", not "tar.gz"
            blkgrp_url = f'https://www2.census.gov/programs-surveys/acs/summary_file/2017/data/5_year_entire_sf/Tracts_Block_Groups_Only.tar'
        # census.gov doesn't accept headerless requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36"}
        with requests.get(blkgrp_url, stream=True, headers = headers) as r:
            local_filename = blkgrp_url.split('/')[-1]
            local_filename = f"{year}_"+ local_filename
            local_filename = config.EXTERNAL_DATA_DIR/"census"/local_filename
            print(local_filename)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f:
                for chunk in (tqdm(r.iter_content(chunk_size=8192), total=total_size,unit='B', unit_scale=True)): 
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        # f.flush()
        return local_filename  

def unzipem(fname, path=config.EXTERNAL_DATA_DIR):
    print(f"Extracting from {fname}...")
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        names = tar.getnames()
        print(names)
        tar.extractall(path=path)
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        names = tar.getnames()
        print(names)
        tar.extractall(path=path)
        tar.close()
    elif(fname.endswith("zip")):
        with zipfile.ZipFile(fname, "r") as zip_ref:
            names = zip_ref.namelist()
            print(names)
            zip_ref.extractall(path=path)
    print(f"{fname} extracted.")


download_census_geo_file(years)
download_census_blkgrp_file(years)

types = ('*.tar.gz', '*.tar', '*.zip') # the tuple of file types
archives = []
for files in types:
    archives.extend(glob.glob(join(config.EXTERNAL_DATA_DIR, files)))

for archive in archives:
    print(archive)
    unzipem(archive)


###TODO: download all, combine into giant h5 file, stream for each year (closest year in the case of 2011 and 2018) and blkgrp. 
### Determine which data is available across the dataset (something like notNan > 90-95%)
### Also cross reference the data identifier numbers with their descriptions and replace, a la what I did for
### diagnosis codes and descriptions

# census_urls = [
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2012/data/5_year_entire_sf/2012_ACS_Geography_Files.zip',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2012/data/5_year_entire_sf/Tracts_Block_Groups_Only.tar.gz',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2013/data/5_year_entire_sf/2013_ACS_Geography_Files.zip',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2013/data/5_year_entire_sf/Tracts_Block_Groups_Only.tar.gz',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2014/data/5_year_entire_sf/2014_ACS_Geography_Files.zip',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2014/data/5_year_entire_sf/Tracts_Block_Groups_Only.tar.gz',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2015/data/5_year_entire_sf/2015_ACS_Geography_Files.zip',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2015/data/5_year_entire_sf/Tracts_Block_Groups_Only.tar.gz',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2016/data/5_year_entire_sf/2016_ACS_Geography_Files.zip',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2016/data/5_year_entire_sf/Tracts_Block_Groups_Only.tar.gz',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2017/data/5_year_entire_sf/2017_ACS_Geography_Files.zip',
# 'https://www2.census.gov/programs-surveys/acs/summary_file/2017/data/5_year_entire_sf/Tracts_Block_Groups_Only.tar', #### no ".gz"
# ]
# def download_file(url):
#     local_filename = url.split('/')[-1]
#     # NOTE the stream=True parameter below
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(local_filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192): 
#                 if chunk: # filter out keep-alive new chunks
#                     f.write(chunk)
#                     # f.flush()
#     return local_filename

# for url in census_urls:
#     download_file(url)


# raw_file = config.GEODATA_RAW
# raw_data = pd.read_pickle(raw_file)
# raw_data = raw_data.reset_index(drop=False)

# file_name = config.GEODATA_BLOCK_INFO
# geo_by_block = pd.read_pickle(file_name)

# result = raw_data.merge(geo_by_block, on="ACS_BlockGroup_geoid", how="left")
# # print(result)

# result_file = config.GEODATA_FINAL
# result.to_pickle(result_file)


# # How long did this take?
# print("This program,", os.path.basename(__file__), "took")
# print(datetime.now() - startTime)
# print("to run.")
