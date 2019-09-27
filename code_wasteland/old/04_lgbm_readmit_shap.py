import sys

sys.path.append("modules")


from datetime import datetime

startTime = datetime.now()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
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

print("Loading model with pickle...")

pkl_model = config.LGBM_READMIT_MODEL_CLASSIFICATION_PICKLE
# load model with pickle to predict
with open(pkl_model, "rb") as fin:
    c_gbm = pickle.load(fin)

c_test_features = pd.read_pickle(config.C_TEST_FEATURES_FILE)
c_test_labels = pd.read_pickle(config.C_TEST_LABELS_FILE)
c_features = pd.read_pickle(config.C_FEATURES_FILE)


print("Generating results tables and figures...")
c_predict_labels = c_gbm.predict(c_test_features)

# predict probabilities
probs = c_predict_labels

fpr, tpr, threshold = metrics.roc_curve(c_test_labels, c_predict_labels)
roc_auc = metrics.auc(fpr, tpr)


# plt.figure(figsize=(5,5))
plt.title("Receiver Operating Characteristic")
plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([-0.011, 1.011])
plt.ylim([-0.011, 1.011])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# plt.savefig(
#     (config.FIGURES_DIR / "Receiver Operating Characteristic AUC %0.2f.png" % roc_auc),
#     dpi=400,
#     transparent=True,
#     bbox_inches="tight",
# )
plt.show()
plt.close()

c_predict_labels = c_gbm.predict(c_test_features)
c_range = c_predict_labels.size

# convert into binary values for classification report
for i in range(0, c_range):
    if c_predict_labels[i] >= 0.15:  # setting threshold to .5
        c_predict_labels[i] = 1
    else:
        c_predict_labels[i] = 0

accuracy = metrics.accuracy_score(c_test_labels, c_predict_labels)
print("Accuracy of GBM classifier: ", accuracy)
print(classification_report_imbalanced(c_test_labels, c_predict_labels))


# How long did this take?
print("This program took")
print(datetime.now() - startTime)
print("to run.")
