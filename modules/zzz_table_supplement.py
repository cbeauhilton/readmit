import pandas as pd
from cbh import config
import glob

re_30 = "readmitted30d"
re_7 = "readmitted7d"
re_5 = "readmitted5d"
re_3 = "readmitted3d"
re_days = "days_between_current_discharge_and_next_admission"

died = "died_within_48_72h_of_admission_combined"

los_3 = "length_of_stay_over_3_days"
los_5 = "length_of_stay_over_5_days"
los_7 = "length_of_stay_over_7_days"
los_days = "length_of_stay_in_days"

target_list = [re_30, re_7, re_5, re_3, 
# re_days, 
# died, 
los_3, los_5, los_7, 
# los_days
]
for target in target_list:
    file_list = glob.glob(f"*_{target}_summary_scores.csv")
    try:
        df = pd.concat((pd.read_csv(f) for f in file_list))
        df["target"].replace(
            {
                re_30: "Readmitted within 30 days",
                re_3: "Readmitted within 3 days",
                re_7: "Readmitted within 7 days",
                re_5: "Readmitted within 5 days",
                re_days: "Days to readmission",
                died: "Death within 48--72 hours",
                los_7: "Hospital stay over 7 days",
                los_5: "Hospital stay over 5 days",
                los_3: "Hospital stay over 3 days",
                los_days: "Length of stay (days)",
            },
            inplace=True,
        )

        df = df.rename(index=str, columns={"target": "Target"},)
        df.set_index("Target", inplace=True)
        print(df.head(10))
        df.to_csv(f"{target}_summary_summary_supplement.csv")

    except:
        pass
    

file_list = glob.glob("*_summary_summary_supplement.csv")
df = pd.concat((pd.read_csv(f) for f in file_list))
df.set_index("Target", inplace=True)
print(df.head(10))
df.to_csv("summary_summary_supplement.csv")
