import sys

sys.path.append("modules")
from datetime import datetime

startTime = datetime.now()

import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import metrics

try:
    import cPickle as pickle
except BaseException:
    import pickle
import config

# import configdocker as config

seed = config.SEED

print("Loading labels and features...")
c_train_labels = pd.read_pickle(config.C_TRAIN_LABELS_FILE)
c_train_features = pd.read_pickle(config.C_TRAIN_FEATURES_FILE)
c_test_labels = pd.read_pickle(config.C_TEST_LABELS_FILE)
c_test_features = pd.read_pickle(config.C_TEST_FEATURES_FILE)
c_valid_labels = pd.read_pickle(config.C_VALID_LABELS_FILE)
c_valid_features = pd.read_pickle(config.C_VALID_FEATURES_FILE)
c_features = pd.read_pickle(config.C_FEATURES_FILE)

print("Loading LightGBM datasets...")
c_d_train = lgb.Dataset(config.LIGHTGBM_READMIT_TRAIN_00)
c_d_test = lgb.Dataset(config.LIGHTGBM_READMIT_TEST_00)
c_d_valid = lgb.Dataset(config.LIGHTGBM_READMIT_VALID_00)

print("Training...")
c_evals_result = {}  # to record eval results for plotting

c_params = config.C_READMIT_PARAMS_LGBM

# c_gbm = lgb.train(config.C_READMIT_TRAIN_OPTIONS)

c_gbm = lgb.train(
    c_params,
    c_d_train,
    num_boost_round=100000,
    valid_sets=c_d_test,
    early_stopping_rounds=100,
    verbose_eval=25,
    evals_result=c_evals_result,
    keep_training_booster=True,
)

print("Dumping model with pickle...")
pkl_model = config.LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE_CV
with open(pkl_model, "wb") as fout:
    pickle.dump(c_gbm, fout)

# To continue training...
# print("Moar training...")
# c_gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10,
#                 init_model=pkl_model,
#                 valid_sets=c_d_test)

# c_gbm.save_model(config.LGBM_READMIT_MODEL_CLASSIFICATION)
# print("Dumping to pickle...")
# with open(pkl_model, "wb") as fout:
#     pickle.dump(c_gbm, fout)
# print("Done.")

# predict probabilities
c_predict_labels = c_gbm.predict(c_test_features)

fpr, tpr, threshold = metrics.roc_curve(c_test_labels, c_predict_labels)
roc_auc = metrics.auc(fpr, tpr)

print("Plot metrics recorded during training...")
ax = lgb.plot_metric(c_evals_result, metric="auc", figsize=(10, 10))
figure_title = "Max AUC %0.2f related to iterations_" % roc_auc
timestr = time.strftime("%Y-%m-%d-%H%M")
ext = ".png"
title = figure_title + timestr + ext
plt.savefig(
    (config.FIGURES_DIR / title), dpi=400, transparent=False, bbox_inches="tight"
)
# plt.show()
plt.close()

# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
