import pandas as pd
from cbh import config


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

df_re_30 = pd.read_csv(f"{re_30}_summary_scores.csv")
df_re_7 = pd.read_csv(f"{re_7}_summary_scores.csv")
df_re_5 = pd.read_csv(f"{re_5}_summary_scores.csv")
df_re_3 = pd.read_csv(f"{re_3}_summary_scores.csv")
# df_re_days = pd.read_csv(f"{re_days}_summary_scores.csv")

# df_died = pd.read_csv(f"{died}_summary_scores.csv")

df_los_7 = pd.read_csv(f"{los_7}_summary_scores.csv")
df_los_5 = pd.read_csv(f"{los_5}_summary_scores.csv")
df_los_3 = pd.read_csv(f"{los_3}_summary_scores.csv")
# df_los_days = pd.read_csv(f"{los_days}_summary_scores.csv")

pdList = []
pdList.extend(value for name, value in locals().items() if name.startswith("df_"))
data = pd.concat(pdList)
data = data.drop(columns="Unnamed: 0")


data["target"].replace(
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

data = data.rename(index=str, columns={"target": "Target"},)
data.set_index('Target', inplace=True)

print(data.head(10))

data.to_csv("summary_summary.csv")
