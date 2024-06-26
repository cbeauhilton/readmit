import os
from datetime import datetime

import pandas as pd
import cbh.config as config

print("About to run", os.path.basename(__file__))
startTime = datetime.now()
seed = config.SEED

#### Make a list of all the features we want to include ####

# target = config.TARGET 
targets = config.LOS_TARGETS

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
    print("Loading", filename)
    data = pd.read_pickle(filename)

    print("File loaded.")

    # final cleaning for LoS prediction
    print("Dropping expired, obs, outpt, ambulatory, emergency patients...")
    data = data[data["dischargedispositiondescription"] != "Expired"]
    data = data[data["patientclassdescription"] != "Observation"]  #
    data = data[data["patientclassdescription"] != "Outpatient"]  # ~10,000
    data = data[
        data["patientclassdescription"] != "Ambulatory Surgical Procedures"
    ]  # ~8,000
    data = data[data["patientclassdescription"] != "Emergency"]  # ~7,000

    print(data["dischargedispositiondescription"].value_counts(dropna=False), "\n")
    print(data["patientclassdescription"].value_counts(dropna=False), "\n")

    #### drop everything but the good stuff ####
    dropem = list(set(list(data)) - set(shap_list))
    data = data.drop(columns=dropem)

    final_file = config.PROCESSED_DATA_DIR / f"{target}.h5"

    print(f"Saving to {final_file}...")
    data.to_hdf(final_file, key=f"{target}clean", mode='a', format='table')

    # sample_size = 300_000
    # print(f"Saving data set with {sample_size} encounters for comparison with Rajkomar...")
    # small_data = data.sample(n=sample_size)
    # small_data.to_hdf(final_file, key=f"{target}cleansmall", mode='a', format='table')