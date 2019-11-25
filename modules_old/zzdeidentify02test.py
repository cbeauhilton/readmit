import hashlib
import os
import random
import string
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import cbh.config as config
import configcols

tqdm.pandas()


try:
    import cPickle as pickle
except BaseException:
    import pickle

print("About to run", os.path.basename(__file__))
startTime = datetime.now()


###############################################################################
#     __                __
#    / /___  ____ _____/ /
#   / / __ \/ __ `/ __  /
#  / / /_/ / /_/ / /_/ /
# /_/\____/\__,_/\__,_/
#
###############################################################################


unscrubbed_file = config.UNSCRUBBED_H5
df = pd.read_hdf(
    unscrubbed_file,
    key="scrubbed",
    columns=["epicdeathdate", "ohiodeathindexdate"],
    start=0,
    stop=10000000,
)

df["died_on_record_0"] = np.where(df["epicdeathdate"].isnull(), "", "dead")
df["died_on_record_1"] = np.where(df["ohiodeathindexdate"].isnull(), "", "dead")
df['died_on_record'] = df['died_on_record_1'].fillna('') + df['died_on_record_0'].fillna('')
# replace field that's entirely space (or empty) with NaN
df = df.replace(r'^\s*$', "no_death_recorded", regex=True)
df = df.replace('deaddead', "dead")
df = df.drop(["died_on_record_1", "died_on_record_0"], axis=1)
print(
    df[
        [
            # "died_on_record_0",
            # "died_on_record_1",
            "died_on_record",
            "epicdeathdate",
            "ohiodeathindexdate",
        ]
    ]
)

# print(df.died_on_record_0.describe())
# print(df.died_on_record_1.describe())
print(df.died_on_record.describe())
