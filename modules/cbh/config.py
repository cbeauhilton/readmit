from pathlib import Path
import os

print("Loading", os.path.basename(__file__))

# Directories
PROJECT_DIR = Path(r"/home/hiltonc/ccf_readmit_1/")
RAW_DATA_DIR = PROJECT_DIR / "data/raw/"
INTERIM_DATA_DIR = PROJECT_DIR / "data/interim/"
EXTERNAL_DATA_DIR = PROJECT_DIR / "data/external/"
PROCESSED_DATA_DIR = PROJECT_DIR / "data/processed/"
SHAP_CSV_DIR = PROJECT_DIR / "data/shap_csv/"
MODELS_DIR = PROJECT_DIR / "models/"

NOTEBOOK_DIR = PROJECT_DIR / "notebooks/"

FIGURES_DIR = PROJECT_DIR / "reports/figures/"
FORCE_PLOT_SINGLETS = FIGURES_DIR / "force_plots/"
PLOTLY_DIR = FIGURES_DIR / "plotly/"

TABLES_DIR = PROJECT_DIR / "reports/tables/"
TABLES_ONE_DIR = TABLES_DIR / "tables_one/"

TEXT_DIR = PROJECT_DIR / "reports/text/pygen/"
TEX_TABLE_DIR = PROJECT_DIR / "reports/text/tables/"

DOCS_DIR = PROJECT_DIR / "docs"
# Specific files

RAW_ZIP_FILE = RAW_DATA_DIR / "ccf_readmit.zip"
RAW_TXT_FILE = RAW_DATA_DIR / "ccf_readmit.txt"

RAW_DATA_FILE_H5 = RAW_DATA_DIR / "ccf_rawdata.h5"
RAW_H5_KEY = "ccf_raw"
# I've grown to prefer pickle files over h5...
RAW_DATA_FILE = RAW_DATA_DIR / "ccf_rawdata.pickle"

GEODATA_RAW_ZIP = EXTERNAL_DATA_DIR / "ANazha_ReadmissionsGeoids_20190219.zip"
GEODATA_RAW = EXTERNAL_DATA_DIR / "geodata_raw.pickle"
GEODATA_BLOCK_INFO = EXTERNAL_DATA_DIR / "geodata_block_info.pickle"
GEODATA_BLOCKS_DONE = EXTERNAL_DATA_DIR / "geodata_blocks_done.csv"
GEODATA_FINAL = EXTERNAL_DATA_DIR / "geodata_final.pickle"
GEODATA_FINAL_CSV = EXTERNAL_DATA_DIR / "geodata_final.csv"
ICD10_DATABASE = EXTERNAL_DATA_DIR / "ICD_10_codes.csv"
DX_CODES_CONVERTED = EXTERNAL_DATA_DIR / "dxcodes.pickle"
CENSUS_API_KEY = "c92c0a70527edc1d91a8c7272af0cb463a4251cf"
# CENSUS_API_KEY = "812c978a2a0d7d99d24fd6e7e22df82b14e1b968"
# CENSUS_API_KEY = "701d6962daf4d31a62672fb901121fd95abcd7c2"
PRETTIFYING_COLUMNS_CSV = PROCESSED_DATA_DIR / "prettifying1.csv"
CCF_CODE_LIST = DOCS_DIR / "value counts/primary_diagnosis_code.csv"
DDW_DIAGNOSES_CSV = DOCS_DIR / "DDW_Diagnoses.csv"

INTERIM_H5 = INTERIM_DATA_DIR / "interim.h5"
INTERIM_PARQ = INTERIM_DATA_DIR / "interim.parq"
CLEAN_PHASE_00 = INTERIM_DATA_DIR / "ccf_clean_phase_00.pickle"
CLEAN_PHASE_01 = INTERIM_DATA_DIR / "ccf_clean_phase_01.pickle"
CLEAN_PHASE_02 = INTERIM_DATA_DIR / "ccf_clean_phase_02.pickle"
CLEAN_PHASE_03 = INTERIM_DATA_DIR / "ccf_clean_phase_03.pickle"
CLEAN_PHASE_04 = INTERIM_DATA_DIR / "ccf_clean_phase_04.pickle"
CLEAN_PHASE_05 = INTERIM_DATA_DIR / "ccf_clean_phase_05.pickle"
CLEAN_PHASE_06 = INTERIM_DATA_DIR / "ccf_clean_phase_06.pickle"
CLEAN_PHASE_07 = INTERIM_DATA_DIR / "ccf_clean_phase_07.pickle"
CLEAN_PHASE_08 = INTERIM_DATA_DIR / "ccf_clean_phase_08.pickle"
CLEAN_PHASE_09 = INTERIM_DATA_DIR / "ccf_clean_phase_09.pickle"
CLEAN_PHASE_10 = INTERIM_DATA_DIR / "ccf_clean_phase_10.pickle"
CLEAN_PHASE_11_TABLEONE = INTERIM_DATA_DIR / "ccf_clean_tableone_phase_11.pickle"

UNSCRUBBED_H5 = INTERIM_DATA_DIR / "readmit_unscrubbed.h5"
SCRUBBED_H5 = INTERIM_DATA_DIR / "readmit_scrubbed.h5"

# change this as needed
# TODO: make selecting the most recent file automatic
PROCESSED_FINAL = CLEAN_PHASE_09
PROCESSED_FINAL_DESCRIPTIVE = CLEAN_PHASE_11_TABLEONE

TRAIN_SET = PROCESSED_DATA_DIR / "train.pickle"
TRAIN_TEST_SET = PROCESSED_DATA_DIR / "train_test.pickle"
UNDERSAMPLE_TRAIN_SET = PROCESSED_DATA_DIR / "undersample_train.pickle"
TEST_SET = PROCESSED_DATA_DIR / "test.pickle"
VALID_SET = PROCESSED_DATA_DIR / "valid.pickle"

TRAINING_REPORTS = TABLES_DIR / "classifiertrainingreports.csv"
REGRESSOR_TRAINING_REPORTS = TABLES_DIR / "regressortrainingreports.csv"
PAPER_NUMBERS = TABLES_DIR / "papernumbers.csv"
RESULTS_TEX = TEX_TABLE_DIR /  "all_results_df.tex"
SCORES_JSON = TABLES_DIR / "scores.json"
SCORES_JSON_SOTA = TABLES_DIR / "scores_sota.json"

# Hyperparameters
NUMERIC_IMPUTER = "median"
SCALER = "StandardScaler"
CATEGORICAL_IMPUTER_STRATEGY = "constant"
CATEGORICAL_IMPUTER_FILL_VALUE = "missing"
ONE_HOT_ENCODER = 1
SEED = 42

LABEL_COLUMN = "readmitted"
READMISSION_THRESHOLD = 30  # number of days

TARGET = "length_of_stay_over_3_days"
# TARGET = "length_of_stay_over_5_days"
# TARGET = "length_of_stay_over_7_days"
# TARGET = "length_of_stay_over_14_days"
NAME_FOR_FIGS = "Length of Stay"

RE_TARGETS = [
"readmitted3d",
"readmitted5d",
"readmitted7d",
"readmitted30d",
]

LOS_TARGETS = [
# "length_of_stay_over_3_days",
# "length_of_stay_over_5_days",
"length_of_stay_over_7_days",
# "length_of_stay_over_14_days",
]

CLASSIFIER_TEST_TARGETS = [
    # "readmitted30d",
    # "length_of_stay_over_3_days",  # median is ~3
    "length_of_stay_over_5_days",  # mean is ~4.5
    # "length_of_stay_over_7_days",
    # "length_of_stay_over_14_days",
    # "financialclass_binary",
    # "gender_binary",
    # "race_binary",
    # "died_within_48_72h_of_admission_combined",
    # "readmitted0_5d",
    # "readmitted1d",
    # "readmitted3d",
    # "readmitted5d",
    # "readmitted7d",
    # "readmitted15d",
    # "readmitted28d",
    # "readmitted45d",
    # "readmitted90d",
    # "readmitted180d",
    # "readmitted365d",
    # "readmitted3650d",
    # "discharged_in_past_30d",
]

C_READMIT_PARAMS_LGBM = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.707630032256903,
    "is_unbalance": "false",
    "learning_rate": 0.010302298912236304,
    "max_depth": -1,
    "min_child_samples": 360,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 568,
    "n_jobs": -1,
    "num_leaves": 99,
    "num_rounds": 10_000_000,
    "objective": "binary",
    "predict_contrib": True,
    "random_state" : SEED,
    "reg_alpha": 0.5926734167821595,
    "reg_lambda": 0.1498749826768534,
    "silent": False,
    "subsample_for_bin": 240000,
    "subsample_freq": 0,
    "subsample": 0.6027609913849075,
    "max_bin": 63,
    "two_round": True
    # "boost_from_average": True,
    # "importance_type": "split",
    # "num_threads": 8,
    # "verbosity": -1
}

REGRESSOR_TEST_TARGETS = [
    "length_of_stay_in_days",
    "patient_age",
    "days_between_current_discharge_and_next_admission",
]

R_READMIT_PARAMS_LGBM = {
    "task": "train",
    "colsample_bytree": 0.6252104665423913,
    "boosting_type": "dart",
    "objective": "regression",
    # "objective": "quantile",
    "metric": {"mae", "mse", "huber", "rmse"},
    "num_leaves": 127,
    "max_depth": -1,
    # "n_estimators": 2192,
    "subsample": 0.8991514172516267,
    "learning_rate": 0.05,
    # "feature_fraction": 0.9,
    # "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 1,
    "min_data_in_bin": 1,
}


##############################################################OLD################################################################

# C_READMIT_DROP_COLS_LGBM = [
#     "admissiontime",
#     "admit_date",
#     "admit_year",
#     "cbc_admit_date",
#     "cbc_discharge_date",
#     "cmp_admit_date",
#     "cmp_discharge_date",
#     "dateofbirth",
#     "days_between_current_admission_and_previous_discharge",
#     "days_between_current_discharge_and_next_admission",
#     "days_since_beginning_of_last_admission",
#     "died_within_48_72h_of_admission_combined",
#     "died_within_48_72h_of_admission_epic",
#     "died_within_48_72h_of_admission_ohio",
#     "died_within_48_72h_of_admission_socialsecurity",
#     "discharge_date",
#     "discharge_year",
#     "dischargetime",
#     "encounterid",
#     "epicdeathdate",
#     "firsticuadmit",
#     "hours_from_admission_to_death_epic",
#     "hours_from_admission_to_death_ohio",
#     "hours_from_admission_to_death_socialsecurity",
#     "lasticudischarge",
#     "lastvisitdate",
#     "Length_of_Stay_over_14_days",
#     "Length_of_Stay_over_5_days",
#     "Length_of_Stay_over_7_days",
#     "length_of_stay",  # length of stay in days is a different column
#     "mrn",
#     "ohiodeathindexdate",
#     "patientid",
#     "socialsecuritydeathdate",
#     "time_between_current_admission_and_previous_discharge",
#     "time_between_current_discharge_and_next_admission",
#     "time_from_admission_to_death_epic",
#     "time_from_admission_to_death_ohio",
#     "time_from_admission_to_death_socialsecurity",
#     "time_since_beginning_of_last_admission",
#     # "admission_hour_of_day",
#     # "admit_day_of_week",
#     # "days_between_current_admission_and_previous_discharge",
#     # "discharge_day_of_week",
#     # "length_of_stay_in_days",
# ]

# C_READMIT_PARAMS_LGBM = {
#     "boosting_type": "dart",
#     'is_unbalance': 'true',
#     # "scale_pos_weight": 5, # about 1/5 readmission rate
#     #'colsample_bytree': 0.6680806918157517,
#     "learning_rate": 0.01,
#     # "min_child_samples": 50,
#     # "num_leaves": 134,
#     # "reg_alpha": 0.03219316044461252,
#     # "reg_lambda": 0.06170436375172991,
#     # "subsample_for_bin": 300000,
#     # "subsample": 0.703381209777314,
#     "metric": {"auc", #"mean_average_precision",
#     },
#     "boost_from_average": True, # adjusts initial score to the mean of labels for faster convergence
#     "verbose": -1,
#     "objective": "binary",
#     "predict_contrib": True,
#     "num_threads": 8,
#     "seed": SEED,

# }

# # Gender
# C_GENDER_LGBM_SHAP_FILE = MODELS_DIR / "shap_values_classification.pickle"

# C_GENDER_LGBM_FULL_DATA = PROCESSED_DATA_DIR / "c_gender_lgbm_full_data.pickle"
# C_GENDER_TRAIN_LABELS_FILE = PROCESSED_DATA_DIR / "c_gender_train_labels.pickle"
# C_GENDER_TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "c_gender_train_features.pickle"
# C_GENDER_TEST_LABELS_FILE = PROCESSED_DATA_DIR / "c_gender_test_labels.pickle"
# C_GENDER_TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "c_gender_test_features.pickle"
# C_GENDER_VALID_LABELS_FILE = PROCESSED_DATA_DIR / "c_gender_valid_labels.pickle"
# C_GENDER_VALID_FEATURES_FILE = PROCESSED_DATA_DIR / "c_gender_valid_features.pickle"
# C_GENDER_FEATURES_FILE = PROCESSED_DATA_DIR / "c_gender_features.pickle"

# # LightGBM doesn't like WindowPaths, so first
# # pull the WindowsPath then resolve it
# # into a string LightGBM can read
# LIGHTGBM_GENDER_READMIT_TRAIN_00 = (
#     PROCESSED_DATA_DIR / "lgbm_gender_readmit_train_00.bin"
# )
# LIGHTGBM_GENDER_READMIT_TEST_00 = PROCESSED_DATA_DIR / "lgbm_gender_readmit_test_00.bin"
# LIGHTGBM_GENDER_READMIT_VALID_00 = (
#     PROCESSED_DATA_DIR / "lgbm_gender_readmit_valid_00.bin"
# )
# LGBM_GENDER_READMIT_MODEL_CLASSIFICATION_PICKLE = (
#     MODELS_DIR / "LGBM_gender_readmit_model_classification.pickle"
# )
# LGBM_GENDER_READMIT_MODEL_REGRESSION_PICKLE = (
#     MODELS_DIR / "LGBM_gender_readmit_model_regression.pickle"
# )

# LIGHTGBM_GENDER_READMIT_TRAIN_00 = str(Path.resolve(LIGHTGBM_GENDER_READMIT_TRAIN_00))
# LIGHTGBM_GENDER_READMIT_TEST_00 = str(Path.resolve(LIGHTGBM_GENDER_READMIT_TEST_00))
# LIGHTGBM_GENDER_READMIT_VALID_00 = str(Path.resolve(LIGHTGBM_GENDER_READMIT_VALID_00))
# LGBM_GENDER_READMIT_MODEL_CLASSIFICATION_PICKLE = str(
#     Path.resolve(LGBM_GENDER_READMIT_MODEL_CLASSIFICATION_PICKLE)
# )
# LGBM_GENDER_READMIT_MODEL_REGRESSION_PICKLE = str(
#     Path.resolve(LGBM_GENDER_READMIT_MODEL_REGRESSION_PICKLE)
# )


# CLEAN_REGRESSION_00 = PROCESSED_DATA_DIR / "ccf_clean_regression_00.h5"
# PROCESSED_H5_PYTORCH = PROCESSED_DATA_DIR / "data_pytorch.h5"

# Train/Test/Validation Split
# TRAIN_START = "2011-01-01"
# TRAIN_END = "2016-12-31"
# TEST_START = "2017-01-01"
# TEST_END = "2017-12-31"
# VALID_START = "2018-01-01"
# VALID_END = "2018-12-31"

# TODO: autorename zip file
# The following doesn't work - at some point, figure out how to
# rename the zip file automagically
# old_file = os.path.join(RAW_DATA_DIR, "*.zip")
# new_file = os.path.join(RAW_DATA_DIR, "ccf_readmit.zip")
# os.rename(old_file, new_file)

# C_LGBM_SHAP_FILE = MODELS_DIR / "shap_values_classification.pickle"
# C_LGBM_SHAP_FILE_NP = MODELS_DIR / "shap_values_classification_numpy"
# C_LGBM_FULL_DATA = PROCESSED_DATA_DIR / "c_lgbm_full_data.pickle"
# C_LGBM_FULL_DATA_CV = PROCESSED_DATA_DIR / "c_lgbm_full_data_cv.pickle"
# C_TRAIN_LABELS_FILE_CV = PROCESSED_DATA_DIR / "c_train_labels_cv.pickle"
# C_TRAIN_FEATURES_FILE_CV = PROCESSED_DATA_DIR / "c_train_features_cv.pickle"
# C_VALID_LABELS_FILE_CV = PROCESSED_DATA_DIR / "c_valid_labels_cv.pickle"
# C_VALID_FEATURES_FILE_CV = PROCESSED_DATA_DIR / "c_valid_features_cv.pickle"
# C_FEATURES_FILE_CV = PROCESSED_DATA_DIR / "c_features_cv.pickle"
# C_TRAIN_LABELS_FILE = PROCESSED_DATA_DIR / "c_train_labels.pickle"
# C_TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "c_train_features.pickle"
# C_TEST_LABELS_FILE = PROCESSED_DATA_DIR / "c_test_labels.pickle"
# C_TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "c_test_features.pickle"
# C_VALID_LABELS_FILE = PROCESSED_DATA_DIR / "c_valid_labels.pickle"
# C_VALID_FEATURES_FILE = PROCESSED_DATA_DIR / "c_valid_features.pickle"
# C_FEATURES_FILE = PROCESSED_DATA_DIR / "c_features.pickle"
# R_TRAIN_LABELS_FILE = PROCESSED_DATA_DIR / "r_train_labels.pickle"
# R_LOS_LGBM_FULL_DATA = PROCESSED_DATA_DIR / "r_los_lgbm_full_data.pickle"
# R_TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "r_train_features.pickle"
# R_TEST_LABELS_FILE = PROCESSED_DATA_DIR / "r_test_labels.pickle"
# R_TEST_FEATURES_FILE = PROCESSED_DATA_DIR / "r_test_features.pickle"
# R_VALID_LABELS_FILE = PROCESSED_DATA_DIR / "r_valid_labels.pickle"
# R_VALID_FEATURES_FILE = PROCESSED_DATA_DIR / "r_valid_features.pickle"
# R_FEATURES_FILE = PROCESSED_DATA_DIR / "r_features.pickle"
# # LightGBM doesn't like WindowPaths, so first
# # pull the WindowsPath then resolve it
# # into a string LightGBM can read
# LIGHTGBM_READMIT_TRAIN_00 = PROCESSED_DATA_DIR / "lgbm_readmit_train_00.bin"
# LIGHTGBM_READMIT_TEST_00 = PROCESSED_DATA_DIR / "lgbm_readmit_test_00.bin"
# LIGHTGBM_READMIT_VALID_00 = PROCESSED_DATA_DIR / "lgbm_readmit_valid_00.bin"
# R_LIGHTGBM_READMIT_TRAIN_00 = PROCESSED_DATA_DIR / "r_lgbm_readmit_train_00.bin"
# R_LIGHTGBM_READMIT_TEST_00 = PROCESSED_DATA_DIR / "r_lgbm_readmit_test_00.bin"
# R_LIGHTGBM_READMIT_VALID_00 = PROCESSED_DATA_DIR / "r_lgbm_readmit_valid_00.bin"
# LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE = (
#     MODELS_DIR / "LGBM_readmit_model_classification.pickle"
# )
# LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE_CV = (
#     MODELS_DIR / "LGBM_readmit_model_classification_cv.pickle"
# )
# LGBM_READMIT_MODEL_REGRESSION_PICKLE = (
#     MODELS_DIR / "LGBM_readmit_model_regression.pickle"
# )
# LIGHTGBM_READMIT_TRAIN_00 = str(Path.resolve(LIGHTGBM_READMIT_TRAIN_00))
# LIGHTGBM_READMIT_TEST_00 = str(Path.resolve(LIGHTGBM_READMIT_TEST_00))
# LIGHTGBM_READMIT_VALID_00 = str(Path.resolve(LIGHTGBM_READMIT_VALID_00))
# R_LIGHTGBM_READMIT_TRAIN_00 = str(Path.resolve(R_LIGHTGBM_READMIT_TRAIN_00))
# R_LIGHTGBM_READMIT_TEST_00 = str(Path.resolve(R_LIGHTGBM_READMIT_TEST_00))
# R_LIGHTGBM_READMIT_VALID_00 = str(Path.resolve(R_LIGHTGBM_READMIT_VALID_00))
# LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE = str(
#     Path.resolve(LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE)
# )
# LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE_CV = str(
#     Path.resolve(LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE_CV)
# )
# LGBM_READMIT_MODEL_REGRESSION_PICKLE = str(
#     Path.resolve(LGBM_READMIT_MODEL_REGRESSION_PICKLE)
# )
