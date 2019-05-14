import sys

sys.path.append("modules")


from datetime import datetime
import time

startTime = datetime.now()

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import z_compare_auc_delong_xu
import shap
import scipy.stats
from scipy import stats
from imblearn.metrics import classification_report_imbalanced
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

try:
    import cPickle as pickle
except BaseException:
    import pickle

import config

seed = config.SEED

print("Loading model from pickle...")

pkl_model = config.LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE
# load model with pickle to predict
with open(pkl_model, "rb") as fin:
    c_gbm = pickle.load(fin)

c_test_features = pd.read_pickle(config.C_TEST_FEATURES_FILE)
c_test_labels = pd.read_pickle(config.C_TEST_LABELS_FILE)
c_features = pd.read_pickle(config.C_FEATURES_FILE)

print("Generating results, tables, and figures...")
# predict probabilities
c_predict_labels = c_gbm.predict(c_test_features)
probs = c_predict_labels

fpr, tpr, threshold = metrics.roc_curve(c_test_labels, c_predict_labels)
roc_auc = metrics.auc(fpr, tpr)

print("Plot feature importances...")
ax = lgb.plot_importance(c_gbm, figsize=(5, 20), importance_type="gain", precision=2)
figure_title = "Feature Importances AUC %0.2f_" % roc_auc
timestr = time.strftime("%Y-%m-%d-%H%M")
ext = ".png"
title = figure_title + timestr + ext
plt.savefig(
    (config.FIGURES_DIR / title), dpi=400, transparent=False, bbox_inches="tight"
)
# plt.show()
plt.close()

print("Generating ROC curve...")
# plt.figure(figsize=(5,5))
plt.title("Receiver Operating Characteristic")
plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([-0.011, 1.011])
plt.ylim([-0.011, 1.011])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
figure_title = "Receiver_Operating_Characteristic_AUC_%0.2f_" % roc_auc
timestr = time.strftime("%Y-%m-%d-%H%M")
ext = ".png"
title = figure_title + timestr + ext
plt.savefig(
    (config.FIGURES_DIR / title), dpi=400, transparent=False, bbox_inches="tight"
)
# plt.show()
plt.close()

print("Generating classification report...")
c_predict_labels = c_gbm.predict(c_test_features)
c_range = c_predict_labels.size

# convert into binary values for classification report
for i in range(0, c_range):
    if c_predict_labels[i] >= 0.15:  # set threshold to desired value
        c_predict_labels[i] = 1
    else:
        c_predict_labels[i] = 0

accuracy = metrics.accuracy_score(c_test_labels, c_predict_labels)
print("Accuracy of GBM classifier: ", accuracy)
print(classification_report_imbalanced(c_test_labels, c_predict_labels))

conf_mx = metrics.confusion_matrix(c_test_labels, c_predict_labels)
# sns.heatmap(conf_mx, square=True, annot=True, fmt='d', cbar=False)

# plt.title('GB Confusion Matrix with Geno-Clinical Features')
# plt.show()

# If you want to save the heatmap to a png, then:
fig = sns.heatmap(conf_mx, square=True, annot=True, fmt="d", cbar=False)
fig = fig.get_figure()
plt.xlabel("True Label")
plt.ylabel("Predicted Label")
figure_title = "Confusion Matrix AUC %0.2f_" % roc_auc
timestr = time.strftime("%Y-%m-%d-%H%M")
ext = ".png"
title = figure_title + timestr + ext
plt.savefig(
    (config.FIGURES_DIR / title), dpi=400, transparent=False, bbox_inches="tight"
)
plt.close()

conf_mx = pd.DataFrame(conf_mx)

FP = conf_mx.sum(axis=0) - np.diag(conf_mx)
FN = conf_mx.sum(axis=1) - np.diag(conf_mx)
TP = np.diag(conf_mx)
TN = conf_mx.values.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
# Specificity or true negative rate
TNR = TN / (TN + FP)
# Precision or positive predictive value
PPV = TP / (TP + FP)
# Negative predictive value
NPV = TN / (TN + FN)
# Fall out or false positive rate
FPR = FP / (FP + TN)
# False negative rate
FNR = FN / (TP + FN)
# False discovery rate
FDR = FP / (TP + FP)

# Overall accuracy
ACC = (TP + TN) / (TP + FP + FN + TN)

print("TPR:", TPR)
print("TNR:", TNR)
print("PPV:", PPV)
print("NPV:", NPV)
print("FPR:", FPR)
print("FNR:", FNR)
print("FDR:", FDR)
print("ACC:", ACC)

# Find 95% Confidence Interval
auc, auc_cov = z_compare_auc_delong_xu.delong_roc_variance(
    c_test_labels, c_predict_labels
)

auc_std = np.sqrt(auc_cov)
alpha = .95
lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

ci[ci > 1] = 1

print("AUC:", auc)
print("AUC COV:", auc_cov)
print("95% AUC CI:", ci)

# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
