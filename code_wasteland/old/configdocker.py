from pathlib import Path

# Directories
PROJECT_DIR = Path("/readmit/")
RAW_DATA_DIR = PROJECT_DIR / "data/raw/"
INTERIM_DATA_DIR = PROJECT_DIR / "data/interim/"
PROCESSED_DATA_DIR = PROJECT_DIR / "data/processed/"
NOTEBOOK_DIR = PROJECT_DIR / "notebooks/"
MODELS_DIR = PROJECT_DIR / "models/"
FIGURES_DIR = PROJECT_DIR / "reports/figures/"

# Specific files
RAW_ZIP_FILE = RAW_DATA_DIR / "ccf_readmit.zip"
RAW_TXT_FILE = RAW_DATA_DIR / "ccf_readmit.txt"

RAW_DATA_FILE = RAW_DATA_DIR / "ccf_rawdata.h5"
RAW_H5_KEY = "ccf_raw"

CLEAN_PHASE_00 = INTERIM_DATA_DIR / "ccf_clean_phase_00.h5"
CLEAN_PHASE_00_KEY = "phase_00"

CLEAN_PHASE_00_PICKLE = INTERIM_DATA_DIR / "ccf_clean_phase_00.pkl"

# change this as needed
PROCESSED_FINAL = CLEAN_PHASE_00
PROCESSED_FINAL_KEY = CLEAN_PHASE_00_KEY

CLEAN_REGRESSION_00 = PROCESSED_DATA_DIR / "ccf_clean_regression_00.h5"
CLEAN_REGRESSION_00_KEY = "clean_regression_00"

TRAIN_TEST_VALID_SPLIT_00 = PROCESSED_DATA_DIR / "ttv_00.h5"
TRAIN_TEST_VALID_SPLIT_00_KEY = 1231235334

PROCESSED_H5_GBM = PROCESSED_DATA_DIR / "c_train_test.h5"
SHAP_FILE = MODELS_DIR / "shap_values_classification.npy"
PROCESSED_H5_PYTORCH = PROCESSED_DATA_DIR / "data_pytorch.h5"

# LightGBM doesn't like WindowPaths, so first
# pull the WindowsPath then resolve it
# into a string LightGBM can read
LIGHTGBM_READMIT_TRAIN_00 = PROCESSED_DATA_DIR / "lgbm_readmit_train_00.bin"
LIGHTGBM_READMIT_TEST_00 = PROCESSED_DATA_DIR / "lgbm_readmit_test_00.bin"
LIGHTGBM_READMIT_VALID_00 = PROCESSED_DATA_DIR / "lgbm_readmit_valid_00.bin"
GBM_MODEL = MODELS_DIR / "GBM_model_classification.txt"

LIGHTGBM_READMIT_TRAIN_00 = str(Path.resolve(LIGHTGBM_READMIT_TRAIN_00))
LIGHTGBM_READMIT_TEST_00 = str(Path.resolve(LIGHTGBM_READMIT_TEST_00))
LIGHTGBM_READMIT_VALID_00 = str(Path.resolve(LIGHTGBM_READMIT_VALID_00))
GBM_MODEL = str(Path.resolve(GBM_MODEL))

# Train/Test/Validation Split
TRAIN_START = "2010-01-01"
TRAIN_END = "2016-12-31"
TEST_START = "2017-01-01"
TEST_END = "2017-12-31"
VALID_START = "2018-01-01"
VALID_END = "2018-12-31"
LABEL_COLUMN = "readmittedany"
READMISSION_THRESHOLD = 30  # number of days

# Hyperparameters
NUMERIC_IMPUTER = "median"
SCALER = "StandardScaler"
CATEGORICAL_IMPUTER_STRATEGY = "constant"
CATEGORICAL_IMPUTER_FILL_VALUE = "missing"
ONE_HOT_ENCODER = 1
SEED = 42
