"""
Retrieve historical US weather data from NOAA.gov
"""

import requests
import pandas as pd
import os
import sys


def get_noaa_data(mytoken, dataset, datatypes, stations, start_date, end_date):
    # Check for API token
    if mytoken == "":
        sys.exit("Missing API token.")
    # Make weather data directories if missing
    if not os.path.isdir("data/external/weather_data"):
        os.mkdir("data/external/weather_data")
    if not os.path.isdir("data/external/weather_data/noaa"):
        os.mkdir("data/external/weather_data/noaa")
    # Define token and URL formatting
    headers = {"token": mytoken}
    url = f"https://www.ncei.noaa.gov/access/services/data/v1?dataset={dataset}&dataTypes={datatypes}&stations={stations}&startDate={start_date}&endDate={end_date}&format=json"
    # Check URL
    print("URL:", url)
    response = requests.get(url, headers=headers)
    if response.status_code == 400:
        print("Status Code 400 = Bad request")
    elif response.status_code == 200:
        print("Status Code 200 = Request OK!")
    elif response.status_code == 500:
        print("Status Code 500 = Internal server error")

    try:
        response = response.json()
        stationData = pd.DataFrame(response)
        print(
            "Successfully retrieved "
            + str(len(stationData["STATION"].unique()))
            + " stations"
        )

        # Convert date to pandas datetime
        stationData["DATE"] = pd.to_datetime(stationData["DATE"])

        # NOAA reports temperatures in 10ths of degrees Celsius
        # Convert from object to numeric then divide by 10
        stationData.TMAX = pd.to_numeric(stationData.TMAX)
        stationData.TMIN = pd.to_numeric(stationData.TMIN)
        stationData.TMAX = stationData.TMAX * 0.1
        stationData.TMIN = stationData.TMIN * 0.1

        # Fill NaN in boolean columns
        stationData.WT08 = stationData.WT08.fillna(0)

        # Convert other numeric columns
        stationData.PRCP = pd.to_numeric(stationData.PRCP)

        # Rename columns (see below for dataype info)
        stationData = stationData.rename(
            index=str,
            columns={
                "WT08": "smoke_or_haze",
                "TMAX": "maximum_temperature_celsius",
                "TMIN": "minimum_temperature_celsius",
                "PRCP": "precipitation",
            },
        )

        print(stationData.head())
        dates = stationData["DATE"]
        print("Last date retrieved: " + str(dates.iloc[-1]))
        stationData.to_pickle(
            f"data/external/weather_data/noaa/{stations}-{start_date}-{end_date}.pickle"
        )
        print("Great success!")
        return stationData

    # Catch all exceptions for a bad request or missing data
    except:
        print("Error converting weather data to dataframe.")


mytoken = "omzahBZzfmkFjYBZxbJYEtshkONUJDhk"
dataset = "daily-summaries"
datatypes = "TMAX,TMIN,PRCP,WT08"
# OK to request datatypes that a given station does not have - will not fail request
stations = "USW00004853"
start_date = "2010-01-01"
end_date = "2019-01-01"

get_noaa_data(mytoken, dataset, datatypes, stations, start_date, end_date)

"""
### STATION IDS ###
# USW00004853 41.5175 -81.6836 178.0 OH CLEVELAND BURKE AP

### DATATYPES ###
# TMAX - Maximum temperature
# TMIN - Minimum temperature
# PRCP - Precipitation
# WT01 - Fog, ice fog, or freezing fog (may include heavy fog)
# WT02 - Heavy fog or heaving freezing fog (not always distinguished from fog)
# WT03 - Thunder
# WT05 - Hail (may include small hail)
# WT08 - Smoke or haze 
# WT10 - Tornado, waterspout, or funnel cloud"

### USEFUL LINKS ###
# https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation (URL formatting for NCEI, NCEI is way easier than the other NOAA resources I tried)
# https://docs.opendata.aws/noaa-ghcn-pds/readme.html (has descriptions of all abbreviations, also citation info)
"""

