import os
from datetime import datetime

import pandas as pd
import cbh.config as config

print("About to run", os.path.basename(__file__))
startTime = datetime.now()
seed = config.SEED

targets = config.RE_TARGETS

for target in targets:
    print(target)
    # Load CSV of top SHAP values, and select the first n
    csv_dir = config.SHAP_CSV_DIR
    shap_file = f"{target}_shap.csv"
    # print("SHAP CSV:", csv_dir / shap_file)
    top_shaps = pd.read_csv(csv_dir / shap_file)
    top_shaps = top_shaps.rename(index=str, columns={"0": "feature_names"})
    shap_index = 22  # [10, 20, 30, 40, 50, 60, 500]
    top_shaps = top_shaps[:shap_index]
    shap_list = top_shaps["feature_names"].tolist()
    shap_list.append(target)  # to make the labels and features sets
    dont_misses = [
        "platelet_count_admit_value",
        "pressureulcer_Present_on_Admission_to_the_Hospital",
        "length_of_stay_in_days",
    ]
    for dont_miss in dont_misses:
        shap_list.append(dont_miss)
    # print(shap_list)


    #### Load data proper ####
    filename = config.PROCESSED_FINAL
    print(f"\n Loading {filename}...")
    data = pd.read_pickle(filename)

    print("File loaded. \n")

    # final cleaning for readmission prediction
    data = data[data["dischargedispositiondescription"] != "Expired"]

    #### drop everything but the good stuff ####
    dropem = list(set(list(data)) - set(shap_list))
    # for i, col in enumerate(dropem):
    #     try:
    #         print(f"Dropping {col}... ({i}/{len(dropem)})")
    #         data = data.drop(columns=col)
    #     except:
    #         print(f"Couldn't drop {col}. Hmm.")

    data = data.drop(columns=dropem)

    data.reset_index(inplace=True)
    data = data.set_index(["index"])
    int_cols = list(data.select_dtypes(include='int').columns)
    for col in int_cols:
        data[col] = data[col].astype("int32")
    data.index = pd.to_numeric(data.index, downcast = 'signed')

    # pkl = config.PROCESSED_DATA_DIR / f"{target}.pkl"
    # data.to_pickle(pkl)

    df = pd.DataFrame(data.to_records(), columns=list(data.columns))
    df = df.loc[:, df.columns.notnull()]

    final_file = config.PROCESSED_DATA_DIR / f"{target}.h5"
    print(f"\n Saving to {final_file}...")
    df.to_hdf(final_file, key=f"{target}clean", mode='a', format='table')
