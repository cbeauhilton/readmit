import pandas as pd
from pathlib import Path
from cbh import config

# filename = config.RAW_DATA_FILE
filename = config.CLEAN_PHASE_09
ccf_data = pd.read_pickle(filename)

filename = Path(r'C:\Users\hiltonc\Desktop\taussig_readmit\data\raw\taussig_rawdata.pickle')
taussig_data = pd.read_pickle(filename)

# print(len(ccf_data))
# print(len(taussig_data))
# print(list(ccf_data))
# print(list(taussig_data))

# stupid unicode characters
taussig_data["patientid"] = taussig_data["ï»¿patientid"]

ccf_ids = ccf_data["patientid"].values.tolist()
# get rid of duplicates
ccf_ids = list(set(ccf_ids))

# different way of doing the same thing
taussig_ids = taussig_data["patientid"].unique().tolist()

# print(len(taussig_ids))

row_list = []
for i, pt in enumerate(taussig_ids):
    if pt in ccf_ids:
        idx = ccf_data[ccf_data["patientid"] == pt].index.tolist()
        row_list.append(idx)
        # print(row_list)
    if i % 100 == 0:
        print(len(taussig_ids) - i)
    else:
        pass
    
    # if i == 4:
    #     break


flat_list = [item for sublist in row_list for item in sublist]


print(len(flat_list))

data = ccf_data[ccf_data.index.isin(flat_list)]

print(list(data))

data.to_csv("taussigtake2.csv")