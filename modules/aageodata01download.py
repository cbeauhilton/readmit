import json
import os
import sys
import zipfile
from datetime import datetime
from io import StringIO
from urllib.request import urlopen
import time

from pathlib import Path

import censusdata
import pandas as pd
from pandas.io.json import json_normalize
from tqdm import tqdm

sys.path.append("modules")
from cbh import config

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.precision", 2)

tqdm.pandas()


print("About to run", os.path.basename(__file__))
startTime = datetime.now()


raw_file = config.GEODATA_RAW
interim_dir = config.INTERIM_DATA_DIR

# 150|00|US|39|035|1164|003
# summary_level| geographic component | country | state# | county | census tract| block group

geo_raw = pd.read_pickle(raw_file)
geo_raw = geo_raw[geo_raw["ACS_BlockGroup_geoid"] != "Not available"]
print("Full length: ", len(geo_raw))
geo_raw = geo_raw.ACS_BlockGroup_geoid.drop_duplicates().tolist()
print("Length after dropping duplicates: ", len(geo_raw))
print("Dropping already complete blocks...")
doneblocks = pd.read_csv(config.GEODATA_BLOCKS_DONE, header=None)
doneblocks = doneblocks[0].tolist()
print("Blocks already pulled: ", len(doneblocks))
geo_raw = [x for x in geo_raw if x not in doneblocks]
print("Blocks left: ", len(geo_raw))


def census_download_convert(
    raw_data
):  # here, "raw_data" will be "geo_raw," the list made just above
    raw_data = raw_data
    api_key = config.CENSUS_API_KEY
    blockgrouplist = raw_data
    downloadcounter = 0 # I found some documentation that says it only allows 1000 requests per hour,
    # but I did >7k with no problems. hm.
    for blockgroupid in tqdm(blockgrouplist):
        summary_level = blockgroupid[0:3]
        geographic_comp = blockgroupid[3:5]
        country = blockgroupid[5:7]
        state = blockgroupid[7:9]
        county = blockgroupid[9:12]
        census_tract = blockgroupid[12:18]
        block_group = blockgroupid[18:]
        print("\n")
        if blockgroupid == "15000US360539404012" or "15000US360659401004": # these appear to not exist, bork the program if included
            continue
        print("Block Group Full ID: ", blockgroupid)
        print("State: ", state)
        print("County: ", county)
        print("Census tract :", census_tract)
        print("Block group :", block_group)
        data = censusdata.download(
            "acs5",
            2016,
            censusdata.censusgeo(
                [
                    ("state", state),
                    ("county", county),
                    ("tract", census_tract),
                    ("block group", block_group),
                ]
            ),
            [
                "B01002_001E",  # : "acs_median_age_total_population",
                "B01002_002E",  # : "acs_median_age_males_total_population",
                "B01002_003E",  # : "acs_median_age_females_total_population",
                "B01003_001E",  # : "acs_total_population_count",
                "B02001_002E",  # : "acs_race_white_alone",
                "B02001_003E",  # : "acs_race_black_alone",
                "B02001_008E",  # : "acs_race_two_or_more",
                "B11001_003E",  # : "acs_married_couple_family",
                "B11001_005E",  # : "acs_male_householder_no_wife_present"
                "B11001_006E",
                "B11001_008E",
                "B11001_009E",
                "B23025_003E",
                "B23025_004E",
                "B23025_005E",
                "B23025_007E",
                "B25010_001E",
                "C17002_002E",  # : "acs_under_50_ratio_income_poverty_level_past_12_mo",
                "C17002_003E",  # : "acs_50_to_99_ratio_income_poverty_level_past_12_mo",
                "C17002_004E",  # : "acs_100_to_124_ratio_income_poverty_level_past_12_mo",
                "C17002_005E",  # : "acs_125_to_149_ratio_income_poverty_level_past_12_mo",
                "C17002_006E",  # : "acs_150_to_184_ratio_income_poverty_level_past_12_mo",
                "C17002_007E",  # : "acs_185_to_199_ratio_income_poverty_level_past_12_mo",
                "C17002_008E",  # : "acs_200_and_over_ratio_income_poverty_level_past_12_mo",
                # "B02001_009E",
                # "B02001_010E",
                # "B05001_002E",  # citizenship_US_US_born
                # "B05001_002E",  # missing when downloaded = delete
                # "B05001_003E", # citizenship_US_PR_Island_born
                # "B05001_004E", # citizenship_US_born_abroad
                # "B05001_005E",  # citizenship_US_naturalized
                # "B05001_005E",  # missing when downloaded = delete
                # "B05001_006E",  # citizenship_not_US
                # "B05001_006E",  # missing when downloaded = delete
                # "B06007_002E",  # missing when downloaded = delete
                # "B06007_002E",  # Speak only English
                # "B06007_004E",  # missing when downloaded = delete
                # "B06007_004E",  # Spanish Speak English "very well"
                # "B06007_005E",  # missing when downloaded = delete
                # "B06007_005E",  # Spanish Speak English less than "very well"
                # "B06007_007E",  # missing when downloaded = delete
                # "B06007_007E",  # Other Speak English "very well"
                # "B06007_008E",  # missing when downloaded = delete
                # "B06007_008E",  # Other Speak English less than "very well"
                # "B06008_002E",
                # "B06008_002E",  # missing when downloaded = delete
                # "B06008_003E",
                # "B06008_003E",  # missing when downloaded = delete
                # "B06008_004E",
                # "B06008_004E",  # missing when downloaded = delete
                # "B06008_005E",
                # "B06008_005E",  # missing when downloaded = delete
                # "B06008_006E",
                # "B06008_006E",  # missing when downloaded = delete
                # "B06009_002E",
                # "B06009_002E",  # missing when downloaded = delete
                # "B06009_003E",
                # "B06009_003E",  # missing when downloaded = delete
                # "B06009_004E",
                # "B06009_004E",  # missing when downloaded = delete
                # "B06009_005E",
                # "B06009_005E",  # missing when downloaded = delete
                # "B06009_006E",
                # "B06009_006E",  # missing when downloaded = delete
                # "B09001_001E",
                # "B09001_001E",  # missing when downloaded = delete
                # "B11017_002E",
                # "B11017_002E",  # missing when downloaded = delete
                # "B11017_003E",
                # "B11017_003E",  # missing when downloaded = delete
                # "B17026_001E",  # "Total_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_002E",  # "50_less_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_003E",  # "50_74_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_004E",  # "75_99_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_005E",  # "100_124_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_006E",  # "125_149_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_007E",  # "150_174_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_008E",  # "175_184_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_009E",  # "185_199_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_010E",  # "200_299_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_011E",  # "300_399_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_012E",  # "400_499_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_013E",  # "500_more_Income_Poverty_Level_Ratio_past_12_mo",
                # "B19001_001E",  # "total_categorical_Household_Income_past_12_months",
                # "B19001_002E",  # "less10k_categorical_Household_Income_past_12_months",
                # "B19001_003E",  # "10_15k_categorical_Household_Income_past_12_months",
                # "B19001_004E",  # "15_20k_categorical_Household_Income_past_12_months",
                # "B19001_005E",  # "20_25k_categorical_Household_Income_past_12_months",
                # "B19001_006E",  # "25_30k_categorical_Household_Income_past_12_months",
                # "B19001_007E",  # "30_35k_categorical_Household_Income_past_12_months",
                # "B19001_008E",  # "35_40k_categorical_Household_Income_past_12_months",
                # "B19001_009E",  # "40_45k_categorical_Household_Income_past_12_months",
                # "B19001_010E",  # "45_50k_categorical_Household_Income_past_12_months",
                # "B19001_011E",  # "50_60k_categorical_Household_Income_past_12_months",
                # "B19001_012E",  # "60_75k_categorical_Household_Income_past_12_months",
                # "B19001_013E",  # "75_99k_categorical_Household_Income_past_12_months",
                # "B19001_014E",  # "100_125k_categorical_Household_Income_past_12_months",
                # "B19001_015E",  # "125_150k_categorical_Household_Income_past_12_months",
                # "B19001_016E",  # "150_200k_categorical_Household_Income_past_12_months",
                # "B19001_017E",  # "more200k_categorical_Household_Income_past_12_months",
                # "B19013_001E",  # "Total_Race_Median_Household_Income_past_12_months",
                # "B19049_001E",  # "Total_Age_Median_Household_Income_past_12_months",
                # "B19049_002E",  # "Under_25_Median_Household_Income_past_12_months",
                # "B19049_003E",  # "25_44_Median_Household_Income_past_12_months",
                # "B19049_004E",  # "45_64_Median_Household_Income_past_12_months",
                # "B19049_005E",  # "Over_65_Median_Household_Income_past_12_months",
                # "B19058_001E",  #  "Total PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS",
                # "B19058_002E",
                # "B19058_002E",  #  "With PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS",
                # "B19058_002E",  # missing when downloaded = delete
                # "B19058_003E",
                # "B19058_003E",  #  "No PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS",
                # "B19058_003E",  # missing when downloaded = delete
                # "B19061_001E",  # "Aggregate_earnings_12mo",
                # "B19080_001E", # "Household_income_Lowest_Quintile",
                # "B19080_002E", # "Household_income_Second_Quintile",
                # "B19080_003E", # "Household_income_Third_Quintile",
                # "B19080_004E", # "Household_income_Fourth_Quintile",
                # "B19080_005E", # "Household_income_Lower_Limit_of_Top_5_Percent",
                # "B19083_001E",
                # "B19083_001E",  # missing when downloaded = delete
                # "B19113_001E",  # "median_family_Income_past_12_months",
                # "B19113_001E",  # missing when downloaded = delete
                # "B25010_002E",
                # "B25010_003E",
                # "B25035_001E",
                # "B25039_001E",
            ],
            key=api_key,  # needed to avoid getting capped at a certain number of downloads
        )
        data = data.rename(
            index=str,
            columns={
                # "B19001_001E": "total_categorical_Household_Income_past_12_months",
                # "B19001_002E": "less10k_categorical_Household_Income_past_12_months",
                # "B19001_003E": "10_15k_categorical_Household_Income_past_12_months",
                # "B19001_004E": "15_20k_categorical_Household_Income_past_12_months",
                # "B19001_005E": "20_25k_categorical_Household_Income_past_12_months",
                # "B19001_006E": "25_30k_categorical_Household_Income_past_12_months",
                # "B19001_007E": "30_35k_categorical_Household_Income_past_12_months",
                # "B19001_008E": "35_40k_categorical_Household_Income_past_12_months",
                # "B19001_009E": "40_45k_categorical_Household_Income_past_12_months",
                # "B19001_010E": "45_50k_categorical_Household_Income_past_12_months",
                # "B19001_011E": "50_60k_categorical_Household_Income_past_12_months",
                # "B19001_012E": "60_75k_categorical_Household_Income_past_12_months",
                # "B19001_013E": "75_99k_categorical_Household_Income_past_12_months",
                # "B19001_014E": "100_125k_categorical_Household_Income_past_12_months",
                # "B19001_015E": "125_150k_categorical_Household_Income_past_12_months",
                # "B19001_016E": "150_200k_categorical_Household_Income_past_12_months",
                # "B19001_017E": "more200k_categorical_Household_Income_past_12_months",
                # "B19013_001E": "Total_Race_Median_Household_Income_past_12_months",
                # "B19049_001E": "Total_Age_Median_Household_Income_past_12_months",
                # "B19049_002E": "Under_25_Median_Household_Income_past_12_months",
                # "B19049_003E": "25_44_Median_Household_Income_past_12_months",
                # "B19049_004E": "45_64_Median_Household_Income_past_12_months",
                # "B19049_005E": "Over_65_Median_Household_Income_past_12_months",
                # "B19061_001E": "Aggregate_earnings_12mo",
                # "B19113_001E": "median_family_Income_past_12_months",
                # "B19301_001": "per_capita_Income_past_12_months",
                # "B20002_002": "male_median_earnings_past_12_months",
                # "B20002_003": "female_median_earnings_past_12_months",
                # "B17026_001E": "Total_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_002E": "50_less_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_003E": "50_74_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_004E": "75_99_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_005E": "100_124_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_006E": "125_149_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_007E": "150_174_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_008E": "175_184_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_009E": "185_199_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_010E": "200_299_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_011E": "300_399_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_012E": "400_499_Income_Poverty_Level_Ratio_past_12_mo",
                # "B17026_013E": "500_more_Income_Poverty_Level_Ratio_past_12_mo",
                # "B19058_001E": "Total PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS",
                # "B19058_002E": "With PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS",
                # "B19058_003E": "No PUBLIC ASSISTANCE INCOME OR FOOD STAMPS/SNAP IN THE PAST 12 MONTHS FOR HOUSEHOLDS",
                # "B19080_001E": "Household_income_Lowest_Quintile",
                # "B19080_002E": "Household_income_Second_Quintile",
                # "B19080_003E": "Household_income_Third_Quintile",
                # "B19080_004E": "Household_income_Fourth_Quintile",
                # "B19080_005E": "Household_income_Lower_Limit_of_Top_5_Percent",
            },
        )
        data[
            "ACS_BlockGroup_geoid"
        ] = blockgroupid  # create a blockgroupid index for concatenating later
        data.reset_index(drop=True)  # this doesn't seem to work?
        data.set_index("ACS_BlockGroup_geoid")  # neither does this?
        file_name = config.GEODATA_BLOCK_INFO
        if (
            file_name.is_file()
        ):  # skip creating file if exists, but must exist to append subsequent loops
            pass
        else:
            data.to_pickle(file_name)
        geo_by_block = pd.read_pickle(file_name)
        geo_by_block = geo_by_block.drop_duplicates()
        geo_by_block = geo_by_block.append(data)
        geo_by_block.to_pickle(file_name)
        print("Pickle file available at", file_name)
        savedoneblocks = open(config.GEODATA_BLOCKS_DONE, "a")
        savedoneblocks.write("{}\n".format(blockgroupid))
        print(f"Block {blockgroupid} finished, see: {config.GEODATA_BLOCKS_DONE}")
        downloadcounter += 1
        print("downloadcounter = ", downloadcounter)
        # if downloadcounter == 1000:
        # time.sleep(60 * 60)
        # downloadcounter == 0


census_download_convert(raw_data=geo_raw)

# def merge_geo_files():
# TODO: make this into a function

raw_file = config.GEODATA_RAW
raw_data = pd.read_pickle(raw_file)
raw_data = raw_data.reset_index(drop=False)

file_name = config.GEODATA_BLOCK_INFO
geo_by_block = pd.read_pickle(file_name)

result = raw_data.merge(geo_by_block, on="ACS_BlockGroup_geoid", how="left")
# print(result)

result_file = config.GEODATA_FINAL
result_csv = config.GEODATA_FINAL_CSV
result.to_pickle(result_file)
result.to_csv(result_csv)


# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")
