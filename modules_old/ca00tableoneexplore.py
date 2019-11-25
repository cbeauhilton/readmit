import os
import sys

sys.path.append("modules")
from datetime import datetime

import pandas as pd
from plotly.offline import iplot
from tableone import TableOne
import time
from cbh import config


print("About to run", os.path.basename(__file__))
startTime = datetime.now()


# display all columns when viewing dataframes: make the number
# anything bigger than your number of columns
pd.options.display.max_columns = 2000

figures_path = config.FIGURES_DIR
tables_path = config.TABLES_ONE_DIR
filename = config.PROCESSED_FINAL_DESCRIPTIVE

print("Loading data file...", filename)
data = pd.read_pickle(filename)
# data.set_index("encounterid", inplace=True)

print("Defining columns...")
columns = [
    "patient_age",
    "gender",
    "race",
    "marital_status",
    "financialclass_orig",  # categorizes insurances from insurance2
    "dischargetime_year",
    "any_cancer_not_skin",
    "metastatic_solid_tumor",
    "solid_organ_transplant",
    "aids_hiv",
    "renal_disease",
    "mild_liver_disease",
    "moderate_or_severe_liver_disease",
    "diabetes_with_chronic_complication",
    "diabetes_without_chronic_complication",
    "hypertension",
    "myocardial_infarction",
    "congestive_heart_failure",
    "cerebrovascular_disease",
    "chronic_pulmonary_disease",
    "pneumonia",
    "dementia",
    "anxiety",
    "depression",
    "psychosis",
    "ondialysis",
    "tpn",
    # "pressureulcer",
    "Low hemoglobin level (<12) at discharge",
    "Low sodium level (<135) at discharge",
    "ed_6month_total",
    "ed_6month",
    "ed_admission",
    "number_past_admissions",
    # "attending_specialty_institute_desc",
    "patientclassdescription",
    "dischargedispositiondescription",
    "days_between_current_admission_and_previous_discharge",
    "readmitted30d",
    "length_of_stay_in_days",
    # "admit_day_of_week",
    # "admit_diastolic_bp",
    # "admit_source_description",
    # "admit_systolic_bp",
    # "admitted_on_holiday",
    # "admitted_on_weekend",
    # "bmi_admit",
    # "cerebral_palsy",
    # "connective_tissue_disorder",
    # "discharged_on_weekend",
    # "epilepsy",
    # "hemiplegia_or_paraplegia",
    # "hip_replacement",
    # "insurance2", # lists all the insurances
    # "knee_replacement",
    # "medsfirst24hours",
    # "medsonadmitdate",
    # "peptic_ulcer_disease",
    # "peripheral_vascular_disease",
    # "primary_language",
    # "rheumatic_disease",
    # "short_gut_syndrome",
]

# columns containing categorical variables
categorical = [
    "dischargetime_year",
    "aids_hiv",
    "anxiety",
    "any_cancer_not_skin",
    # "attending_specialty_institute_desc",
    "cerebrovascular_disease",
    "chronic_pulmonary_disease",
    "congestive_heart_failure",
    "dementia",
    "depression",
    "marital_status",
    "diabetes_with_chronic_complication",
    "diabetes_without_chronic_complication",
    "dischargedispositiondescription",
    "ed_6month",
    "ed_admission",
    "financialclass_orig",
    "gender",
    "hypertension",
    "Low hemoglobin level (<12) at discharge",
    "Low sodium level (<135) at discharge",
    "metastatic_solid_tumor",
    "mild_liver_disease",
    "moderate_or_severe_liver_disease",
    "myocardial_infarction",
    "ondialysis",
    "patientclassdescription",
    "pneumonia",
    # "pressureulcer",
    "psychosis",
    "race",
    "readmitted30d",
    "renal_disease",
    "solid_organ_transplant",
    "tpn",
    # 'hospital_transfer_description',
    # "admit_day_of_week",
    # "admit_source_description",
    # "admitted_on_holiday",
    # "admitted_on_weekend",
    # "cerebral_palsy",
    # "connective_tissue_disorder",
    # "discharged_on_weekend",
    # "epilepsy",
    # "hemiplegia_or_paraplegia",
    # "hip_replacement",
    # "insurance2",
    # "knee_replacement",
    # "marital_status",
    # "peptic_ulcer_disease",
    # "peripheral_vascular_disease",
    # "primary_language",
    # "rheumatic_disease",
    # "short_gut_syndrome",
]

# non-normal variables (when you generate the TableOne it will tell you
# at the bottom which variables it thinks are non-normal)

nonnormal = [
    "bmi_admit",
    "days_between_current_admission_and_previous_discharge",
    "ed_6month_total",
    "length_of_stay_in_days",
    "patient_age",
    "number_past_admissions",
]

# set decimal places
# decimals = {"ageexactyrs": 0, 'hemoglobin_discharge': 0, 'sodium_discharge': 0}
decimals = {}

# prettify labels
labels = {
    "dischargetime_year": "Year of Admission",
    "aids_hiv": "AIDS/HIV",
    "anxiety": "Anxiety",
    "any_cancer_not_skin": "Cancer",
    # "attending_specialty_institute_desc": "Hospital service",
    "bmi_admit": "BMI at Admission",
    "cerebrovascular_disease": "Cerebrovascular disease",
    "chronic_pulmonary_disease": "COPD",
    "congestive_heart_failure": "CHF",
    "days_between_current_admission_and_previous_discharge": "Days since last discharge",
    "dementia": "Dementia",
    "depression": "Depression",
    "diabetes_with_chronic_complication": "Diabetes with chronic complication",
    "diabetes_without_chronic_complication": "Diabetes without chronic complication",
    "dischargedispositiondescription": "Discharge location",
    "ed_6month_total": "Total ED visits in the last 6 months",
    "ed_6month": "Number of patients with any ED visits in past 6 months",
    "ed_admission": "Admitted from the ED",
    "financialclass_orig": "Financial Class",  # categorizes insurances from insurance2
    "gender": "Gender",
    "hypertension": "Hypertension",
    "length_of_stay_in_days": "Length of stay in days",
    "marital_status": "Marital status",
    "metastatic_solid_tumor": "Metastatic solid tumor",
    "mild_liver_disease": "Mild liver disease",
    "moderate_or_severe_liver_disease": "Moderate or severe liver disease",
    "myocardial_infarction": "Myocardial infarction",  # history or admission dx?
    "number_past_admissions": "Previous hospitalizations",
    "ondialysis": "Receiving dialysis",
    "patient_age": "Age",
    "patientclassdescription": "Admission class",
    "pneumonia": "Pneumonia",
    # "pressureulcer": "Pressure ulcer",
    "psychosis": "Psychosis",
    "race": "Race/Ethnicity",
    "readmitted30d": "30-day readmissions",
    "renal_disease": "Renal disease",
    "solid_organ_transplant": "Solid organ transplant",
    "tpn": "On total parenteral nutrition before or during admission",  # at baseline or during admission?
}

# categorical variable for stratification
groupbys = [
    "length_of_stay_over_7_days",
    "",
    "readmitted30d",
    # "dischargetime_year",
    # "any_cancer_not_skin",
    # "gender",
    # "race",
    # "readmitted7d",
    # "length_of_stay_over_5_days",
    # "financialclass_orig",
    # "attending_specialty_institute_desc",
]

final_table = ["length_of_stay_over_7_days", "", "readmitted30d"]

for groupby in groupbys:
    # create tableone with the input arguments
    print(f"Creating Table One for {groupby}!..")
    mytable = TableOne(
        data,
        groupby=groupby,
        columns=columns,
        categorical=categorical,
        nonnormal=nonnormal,
        labels=labels,
        label_suffix=True,
        decimals=decimals,
        isnull=False,
        pval=False,
    )
    if groupby == "length_of_stay_over_7_days":
        print("Dropping obs for LOS...")
        
        los_data = data[data["dischargedispositiondescription"] != "Expired"]
        los_data = los_data[los_data["patientclassdescription"] != "Observation"]  #
        los_data = los_data[los_data["patientclassdescription"] != "Outpatient"]  # ~10,000
        los_data = los_data[
            los_data["patientclassdescription"] != "Ambulatory Surgical Procedures"
        ]  # ~8,000
        los_data = los_data[los_data["patientclassdescription"] != "Emergency"]  # ~7,000
        mytable = TableOne(
                los_data,
                groupby=groupby,
                columns=columns,
                categorical=categorical,
                nonnormal=nonnormal,
                labels=labels,
                label_suffix=True,
                decimals=decimals,
                isnull=False,
                pval=False,
            )
    print("Saving to file...")
    figure_title = "_tableone_"
    timestr = time.strftime("%Y-%m-%d-%H%M")
    title = groupby + figure_title + timestr
    # Save table to LaTeX
    fn = os.path.join(tables_path, title)
    mytable.to_latex(fn + ".tex")
    # Save table to HTML
    mytable.to_html(fn + ".html")
    print(f"Table one for {groupby} saved to LaTeX and HTML.")
    if groupby in final_table:
        # Save without date for easy merging, overwriting
        title1 = groupby + figure_title
        fn = os.path.join(tables_path, title1)
        # Save table to LaTeX
        mytable.to_latex(fn + ".tex")
        # Save table to HTML
        mytable.to_html(fn + ".html")
        # Save table to CSV
        mytable.to_csv(fn + ".csv")
        print(f"Table one for {groupby} saved to LaTeX, HTML, and CSV.")


# Display
# mytable


# The to_latex command is not quite plug-and-play.

# Using Tex Live via Atom/VSCode (whatever) Latex package, make sure you have the following to get a usable PDF:

# \documentclass{article}
# \usepackage{booktabs, adjustbox}
# \usepackage[T1]{fontenc}
# \usepackage{lmodern}
# \begin{document}
# \begin{adjustbox}{width={\textwidth},totalheight={\textheight},keepaspectratio}%
# \begin{tabular}{llllll}
# the number of "l"s is autogenerated. It's the number of left-justified columns.

# and at the bottom:

# \bottomrule
# \end{tabular}
# \end{adjustbox}
# \end{document}

print("Saving overall descriptive stats to file...")
# Save descriptive stats to csv file
descript = os.path.join(tables_path, "descriptivestats")
df4 = data.describe()
df4.to_csv(descript + ".csv")


print("Done.")


# How long did this take?
print("This program,", os.path.basename(__file__), "took")
print(datetime.now() - startTime)
print("to run.")


################################################################################

# for groupby in groupbys:
#     # create tableone with the input arguments
#     if groupby == "readmitted30d":
#         data = data[data["dischargedispositiondescription"] != "Expired"]
#         data = data[
#             data["dischargedispositiondescription"] != "General Acute Care Hospital"
#         ]
#         data = data[data["admissiontime_year"] != 2011]
#         data = data[data["readmitted30d"].notnull()]
#     print(f"Creating Table One for {groupby}!..")
#     mytable = TableOne(
#         data,
#         groupby=groupby,
#         columns=columns,
#         categorical=categorical,
#         nonnormal=nonnormal,
#         labels=labels,
#         label_suffix=True,
#         decimals=decimals,
#         isnull=False,  # shows number of "isnull" for each feature
#         pval=False,
#     )

#     print("Saving to file...")
#     figure_title = "_tableone_"
#     timestr = time.strftime("%Y-%m-%d-%H%M")
#     title = groupby + figure_title + timestr
#     # Save table to LaTeX
#     fn = os.path.join(tables_path, title)
#     mytable.to_latex(fn + ".tex")
#     # Save table to HTML
#     mytable.to_html(fn + ".html")
#     # Save table to CSV
#     mytable.to_csv(fn + ".csv")
#     print(f"Table one for {groupby} saved to LaTeX, HTML, and CSV.")

# Display
# mytable
