import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.callbacks import *
from fastai.tabular import *
from matplotlib.colors import ListedColormap
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb

from cbh import config
from cbh.generalHelpers import get_latest_folders, load_ttv_split

target = "length_of_stay_over_5_days"

modelfolder, datafolder, figfolder, tablefolder = get_latest_folders(target)

(
    train_features,
    train_labels,
    test_features,
    test_labels,
    valid_features,
    valid_labels,
    labels,
    features,
) = load_ttv_split(modelfolder)


# set up fastai
cont_names = [
    "days_between_current_admission_and_previous_discharge",
    "heartrate_admit",
    "length_of_stay_of_last_admission",
    "medsfirst24hours",
    "patient_age",
    "temperature_admit",
]
cat_names = [name for name in list(features) if name not in cont_names]
dep_var = target
# fastai expects the dataframe to contain the label
features[target] = labels

features.to_csv("tmp.csv")

valid_idx = range(len(features) - 100000, len(features))


procs = [FillMissing, Categorify, Normalize]
data = TabularDataBunch.from_df(
    path=modelfolder,
    df=features,
    dep_var=dep_var,
    valid_idx=valid_idx,
    procs=procs,
    cat_names=cat_names,
    cont_names=cont_names,
    bs=16382,
)

# print(data.train_ds.cont_names)
# print(data.train_ds.cat_names)

mcc = MatthewsCorreff()
cm = ConfusionMatrix()
auroc = AUROC()
rec = Recall()
prec = Precision()

learn = tabular_learner(
    data,
    layers=[200, 100],
    metrics=[accuracy, mcc, auroc, rec, prec],
    model_dir="fastai_models",
    callback_fns=[CSVLogger],

)

## force GPU activation
# learn.model = learn.model.cuda()
# learn.lr_find()
# learn.recorder.plot(return_fig=True)
# plt.savefig(figfolder / "learner_find.pdf", bbox_inches="tight")
# print(figfolder / f"{target}_learner_find.pdf")
# plt.close()


# learn.fit_one_cycle(10, 1e-2, callbacks=[SaveModelCallback(learn)])
# learn.recorder.plot(return_fig=True)
# plt.savefig(figfolder / "learner.pdf", bbox_inches="tight")
# print(figfolder / f"{target}_learner.pdf")
# plt.close()

learn.load('bestmodel')

y_pred, y_test, loss = learn.get_preds(with_loss=True)



y_pred = [prob[1] for prob in y_pred.tolist()]
# print(y_pred[:5], "\n", y_test[:5], "\n", loss[:5])

cls_name = "Deep Neural Network"
scores = {}
scores[cls_name] = {}

scoring_funcs = [(average_precision_score(y_test, y_pred), f"{average_precision_score.__name__}"),
    (brier_score_loss(y_test, y_pred), f"{brier_score_loss.__name__}"), # roc_auc_score
    (roc_auc_score(y_test, y_pred), f"{roc_auc_score.__name__}"),
    ]

for score, func_name in scoring_funcs:
    scores[cls_name][func_name] = score

# Set up calibration curve
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")


fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_pred, n_bins=10
)

ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (cls_name,))

ax2.hist(y_pred, range=(0, 1), bins=10, label=cls_name, histtype="step", lw=2)


# convert into binary values for classification report
for i in range(len(y_pred)):
    if (
        y_pred[i] >= 0.5
    ):  # set threshold to desired value
        y_pred[i] = 1
    else:
        y_pred[i] = 0

scoring_funcs = [
    (accuracy_score(y_test, y_pred), f"{accuracy_score.__name__}"),
    (f1_score(y_test, y_pred), f"{f1_score.__name__}"),
    (precision_score(y_test, y_pred), f"{precision_score.__name__}"),
    (recall_score(y_test, y_pred), f"{recall_score.__name__}"),
    (matthews_corrcoef(y_test, y_pred), f"{matthews_corrcoef.__name__}"),
]

for score, func_name in scoring_funcs:
    scores[cls_name][func_name] = score

# print(scores)

features = features.drop(target, axis=1)
os.remove("tmp.csv")

#####################################################################################################################

X_train = train_features
y_train = train_labels
X_test = test_features
y_test = test_labels

################################################################################################################

# LGBM

cls_name = "Gradient Boosting Machine"

# set training params
class_thresh = 0.5
params = config.C_READMIT_PARAMS_LGBM
gbm_model = lgb.LGBMClassifier(**params)

early_stopping_rounds = 200
from sklearn.exceptions import UndefinedMetricWarning
# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

gbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="logloss",
    early_stopping_rounds=early_stopping_rounds,
)

y_pred = gbm_model.predict_proba(X_test)[:, 1]
    
scores[cls_name] = {}

scoring_funcs = [(average_precision_score(y_test, y_pred), f"{average_precision_score.__name__}"),
    (brier_score_loss(y_test, y_pred), f"{brier_score_loss.__name__}"), # roc_auc_score
    (roc_auc_score(y_test, y_pred), f"{roc_auc_score.__name__}"),
    ]

for score, func_name in scoring_funcs:
    scores[cls_name][func_name] = score


fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_pred, n_bins=10
)

ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (cls_name,))

ax2.hist(y_pred, range=(0, 1), bins=10, label=cls_name, histtype="step", lw=2)


# convert into binary values for classification report
for i in range(len(y_pred)):
    if (
        y_pred[i] >= 0.5
    ):  # set threshold to desired value
        y_pred[i] = 1
    else:
        y_pred[i] = 0

scoring_funcs = [
    (accuracy_score(y_test, y_pred), f"{accuracy_score.__name__}"),
    (f1_score(y_test, y_pred), f"{f1_score.__name__}"),
    (precision_score(y_test, y_pred), f"{precision_score.__name__}"),
    (recall_score(y_test, y_pred), f"{recall_score.__name__}"),
    (matthews_corrcoef(y_test, y_pred), f"{matthews_corrcoef.__name__}"),
]

for score, func_name in scoring_funcs:
    scores[cls_name][func_name] = score


################################################################################################################

# Everything else

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

numeric_features = features.select_dtypes(include=["int64", "float64"]).columns
categorical_features = features.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

classifiers = [
    ("Logistic Regression", LogisticRegression()),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("Complement Naive Bayes", ComplementNB()),
    ("Support Vector Machine", LinearSVC()),
    ("Stochastic Gradient Descent", SGDClassifier()),
    # ("knn", KNeighborsClassifier(3, n_jobs=-1)),
    # mlp = (MLPClassifier(alpha=1, max_iter=1000),)

]



for cls_name, classifier in classifiers:
    print(f"\n {cls_name} training...")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    except:
        y_pred_proba = y_pred

    scoring_funcs = [
        (average_precision_score(y_test, y_pred_proba), f"{average_precision_score.__name__}"), # proba
        (brier_score_loss(y_test, y_pred_proba), f"{brier_score_loss.__name__}"), # proba
        (roc_auc_score(y_test, y_pred_proba), f"{roc_auc_score.__name__}"), # proba
        (accuracy_score(y_test, y_pred), f"{accuracy_score.__name__}"),
        (f1_score(y_test, y_pred), f"{f1_score.__name__}"),
        (precision_score(y_test, y_pred), f"{precision_score.__name__}"),
        (recall_score(y_test, y_pred), f"{recall_score.__name__}"),
        (matthews_corrcoef(y_test, y_pred), f"{matthews_corrcoef.__name__}"),
    ]
    scores[cls_name] = {}

    for score, func_name in scoring_funcs:
        scores[cls_name][func_name] = score

    # print(scores)

    if hasattr(classifier, "predict_proba"):
        prob_pos = pipe.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = pipe.decision_function(X_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, prob_pos, n_bins=10
    )

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (cls_name,))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=cls_name, histtype="step", lw=2)

print(scores)

with open(config.SCORES_JSON_SOTA, "w") as f:
    json.dump(scores, f, indent=4)


ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title("Calibration plots (reliability curve)")

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.savefig(figfolder/"sota_calibration.pdf")
plt.close()


