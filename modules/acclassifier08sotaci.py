import glob
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import mlxtend
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.exceptions import UndefinedMetricWarning
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

from cbh import config
from cbh.generalHelpers import get_latest_folders, load_ttv_split
from cbh.lgbmHelpers import bootstrap_estimate_and_ci
from cbh.plottingHelpers import save_pr_curve

# import lightgbm as lgb
# from fastai.callbacks import *
# from fastai.tabular import *

# When training starts, certain metrics are often zero for a while, which throws a warning and clutters the terminal output
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)

##################################################################################################################

target = config.TARGET
print(target)

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

X_train = train_features
y_train = train_labels
X_test = test_features
y_test = test_labels
X = features
y = labels


# grab desired scoring functions
scoring_funcs_0 = [average_precision_score, roc_auc_score]
scoring_funcs_1 = [
    brier_score_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
]

n_splits = 5  # number of splits for bootstrap
method = ".632"  # type of bootstrap evaluation method

subsample_n = len(X)
random_state = config.SEED

idx = np.random.choice(np.arange(len(X)), subsample_n, replace=False)
X_boot = X.iloc[idx]
y_boot = y.iloc[idx]

# Set up calibrationFig. S1a curve
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

# initialize scores dictionary
scores = {}

################################################################################################################

# LGBM

# cls_name = "Gradient Boosting Machine"
# scores[cls_name] = {}
# bootstrap = True
# # train new models or load saved
# train_or_load = "train"

# # set training params
# class_thresh = 0.5
# params = config.C_READMIT_PARAMS_LGBM
# gbm_model = lgb.LGBMClassifier(**params)

# early_stopping_rounds = 200


# if train_or_load == "train":

#     gbm_model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_test, y_test)],
#         eval_metric="logloss",
#         early_stopping_rounds=early_stopping_rounds,
#     )

#     ax = lgb.plot_importance(gbm_model, max_num_features=12, grid=False)
#     plt.savefig(f"{figfolder}/{target}_feature_importance.pdf")
#     plt.close()
#     t = time.localtime()
#     current_time = time.strftime("%H%M", t)
#     pkl_model = modelfolder / f"{target}_{cls_name}_MODEL_{current_time}.pkl"
#     with open(pkl_model, "wb") as f:
#         pickle.dump(gbm_model, f)

# if bootstrap:
#     ci_dict_1 = bootstrap_estimate_and_ci(
#         gbm_model,
#         X_boot,
#         y_boot,
#         scoring_func=scoring_funcs_1,
#         method=method,
#         n_splits=n_splits,
#     )

#     cloned_estimator = clone(gbm_model)
#     cloned_estimator.predict = cloned_estimator.decision_function

#     ci_dict_0 = bootstrap_estimate_and_ci(
#         cloned_estimator,
#         X_boot,
#         y_boot,
#         scoring_func=scoring_funcs_0,
#         method=method,
#         n_splits=n_splits,
#     )

#     scores[cls_name] = ci_dict_1.update(ci_dict_0)

# modeldir = f"{str(modelfolder)}/"

# # grab the newest model file from the directory
# pkl_model = max(glob.iglob(modeldir + "*MODEL*"), key=os.path.getmtime)

# with open(pkl_model, "rb") as fin:
#     gbm_model = pickle.load(fin)


# y_pred = gbm_model.predict_proba(X_test)[:, 1]

# if not bootstrap:
#     scoring_funcs = [
#         (
#             average_precision_score(y_test, y_pred),
#             f"{average_precision_score.__name__}",
#         ),
#         (
#             brier_score_loss(y_test, y_pred),
#             f"{brier_score_loss.__name__}",
#         ),  # roc_auc_score
#         (roc_auc_score(y_test, y_pred), f"{roc_auc_score.__name__}"),
#     ]

#     for score, func_name in scoring_funcs:
#         scores[cls_name][func_name] = score

# save_pr_curve(
#     target=target,
#     classifier=cls_name,
#     test_labels=y_test,
#     predicted_labels=y_pred,
#     figfolder=figfolder,
# )

# fraction_of_positives, mean_predicted_value = calibration_curve(
#     y_test, y_pred, n_bins=10
# )

# ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (cls_name,))

# ax2.hist(y_pred, range=(0, 1), bins=10, label=cls_name, histtype="step", lw=2)

# if not bootstrap:
#     # convert into binary values for classification report
#     for i in range(len(y_pred)):
#         if y_pred[i] >= 0.5:  # set threshold to desired value
#             y_pred[i] = 1
#         else:
#             y_pred[i] = 0

#     scoring_funcs = [
#         (accuracy_score(y_test, y_pred), f"{accuracy_score.__name__}"),
#         (f1_score(y_test, y_pred), f"{f1_score.__name__}"),
#         (precision_score(y_test, y_pred), f"{precision_score.__name__}"),
#         (recall_score(y_test, y_pred), f"{recall_score.__name__}"),
#         (matthews_corrcoef(y_test, y_pred), f"{matthews_corrcoef.__name__}"),
#     ]

#     for score, func_name in scoring_funcs:
#         scores[cls_name][func_name] = score

#     print(scores)
##################################################################################################################

# set up fastai

# train model or load saved model
# options = "train" to train, anything else to load
# train_or_load = "train"

# cont_names = [
#     "days_between_current_admission_and_previous_discharge",
#     "heartrate_admit",
#     "length_of_stay_of_last_admission",
#     "medsfirst24hours",
#     "patient_age",
#     "temperature_admit",
# ]

# cont_names = list(features.select_dtypes(include=[np.number]).columns.values)
# cat_names = [name for name in list(features) if name not in cont_names]
# dep_var = target

# # fastai expects the dataframe to contain the label
# features[target] = labels

# features.to_csv("tmp.csv")

# valid_idx = range(len(features) - 100000, len(features))

# procs = [FillMissing, Categorify, Normalize]
# data = TabularDataBunch.from_df(
#     path=modelfolder,
#     df=features,
#     dep_var=dep_var,
#     valid_idx=valid_idx,
#     procs=procs,
#     cat_names=cat_names,
#     cont_names=cont_names,
#     bs=16382,
# )

# # print(data.train_ds.cont_names)
# # print(data.train_ds.cat_names)

# mcc = MatthewsCorreff()
# cm = ConfusionMatrix()
# auroc = AUROC()
# rec = Recall()
# prec = Precision()

# learn = tabular_learner(
#     data,
#     layers=[200, 100],
#     metrics=[accuracy, mcc, auroc, rec, prec],
#     model_dir="fastai_models",
#     callback_fns=[CSVLogger],
# )


# if train_or_load == "train":
#     # force GPU activation
#     learn.model = learn.model.cuda()
#     learn.lr_find()
#     learn.recorder.plot(return_fig=True)
#     plt.savefig(figfolder / "learner_find.pdf", bbox_inches="tight")
#     print(figfolder / f"{target}_learner_find.pdf")
#     plt.close()


#     learn.fit_one_cycle(10, 1e-2, callbacks=[SaveModelCallback(learn)])
#     learn.recorder.plot(return_fig=True)
#     plt.savefig(figfolder / "learner.pdf", bbox_inches="tight")
#     print(figfolder / f"{target}_learner.pdf")
#     plt.close()

# learn.load("bestmodel")

# y_pred, y_test_, loss = learn.get_preds(with_loss=True)
# y_pred = [prob[1] for prob in y_pred.tolist()]
# # print(y_pred[:5], "\n", y_test[:5], "\n", loss[:5])

# cls_name = "Deep Neural Network"
# scores[cls_name] = {}

# scoring_funcs = [
#     (average_precision_score(y_test_, y_pred), f"{average_precision_score.__name__}"),
#     (brier_score_loss(y_test_, y_pred), f"{brier_score_loss.__name__}"),  # roc_auc_score
#     (roc_auc_score(y_test_, y_pred), f"{roc_auc_score.__name__}"),
# ]

# for score, func_name in scoring_funcs:
#     scores[cls_name][func_name] = score

# save_pr_curve(target=target, classifier=cls_name, test_labels=y_test_, predicted_labels=y_pred, figfolder=figfolder)

# fraction_of_positives, mean_predicted_value = calibration_curve(
#     y_test_, y_pred, n_bins=10
# )

# ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (cls_name,))

# ax2.hist(y_pred, range=(0, 1), bins=10, label=cls_name, histtype="step", lw=2)


# # convert into binary values for classification report
# for i in range(len(y_pred)):
#     if y_pred[i] >= 0.5:  # set threshold to desired value
#         y_pred[i] = 1
#     else:
#         y_pred[i] = 0

# scoring_funcs = [
#     (accuracy_score(y_test_, y_pred), f"{accuracy_score.__name__}"),
#     (f1_score(y_test_, y_pred), f"{f1_score.__name__}"),
#     (precision_score(y_test_, y_pred), f"{precision_score.__name__}"),
#     (recall_score(y_test_, y_pred), f"{recall_score.__name__}"),
#     (matthews_corrcoef(y_test_, y_pred), f"{matthews_corrcoef.__name__}"),
# ]

# for score, func_name in scoring_funcs:
#     scores[cls_name][func_name] = score


# features = features.drop(target, axis=1)
# os.remove("tmp.csv")

#####################################################################################################################

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
    ("Gradient Boosting Machine", HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=10_000_000,
            max_leaf_nodes=99,
            min_samples_leaf=360,
            l2_regularization=0.149874,
            max_bins=63,
            warm_start=False,
            scoring='loss',
            validation_fraction=0.1,
            n_iter_no_change=200,
            tol=1e-07,
            verbose=1,
            random_state=config.SEED,
        )), #train loss: 0.54731, val loss: 0.59599
    ("Logistic Regression", LogisticRegression(max_iter=500)),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("Complement Naive Bayes", ComplementNB()),
    ("Support Vector Machine", LinearSVC()),
    # ("Stochastic Gradient Descent", SGDClassifier()),
    # ("knn", KNeighborsClassifier(3, n_jobs=-1)),
    # mlp = (MLPClassifier(alpha=1, max_iter=1000),)
]


for cls_name, classifier in classifiers:
    print(f"\n {cls_name} training...")
    # if cls_name =="Gradient Boosting Machine":
        # pipe = classifier
    # else:
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    except:
        y_pred_proba = y_pred

    save_pr_curve(
        target=target,
        classifier=cls_name,
        test_labels=y_test,
        predicted_labels=y_pred,
        figfolder=figfolder,
    )

    if hasattr(classifier, "predict_proba"):
        prob_pos = pipe.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = pipe.decision_function(X_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, prob_pos, n_bins=10
    )

    ax1.plot(
        mean_predicted_value, fraction_of_positives, "s-", label=f"{cls_name}"
    )

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=cls_name, histtype="step", lw=2)

    ci_dict = {}
    ci_dict_1 = bootstrap_estimate_and_ci(
        pipe,
        X_boot,
        y_boot,
        scoring_func=scoring_funcs_1,
        n_splits=n_splits,
        method=method,
    )

    cloned_estimator = clone(classifier)
    if hasattr(classifier, "predict_proba"):
        cloned_estimator.predict = cloned_estimator.predict_proba
    else:  # use decision function
        cloned_estimator.predict = cloned_estimator.decision_function

    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", cloned_estimator)]
    )

    ci_dict_0 = bootstrap_estimate_and_ci(
        pipe,
        X_boot,
        y_boot,
        scoring_func=scoring_funcs_0,
        n_splits=n_splits,
        method=method,
    )

    ci_dict.update(ci_dict_0)
    ci_dict.update(ci_dict_1)

    scores[cls_name] = ci_dict
    print(scores)

print(scores)

with open(f"{config.TABLES_DIR}/{target}_scores.json", "w") as f:
    json.dump(scores, f, indent=4)


ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title("Calibration plots (reliability curve)")

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.savefig(figfolder / f"{target}_sota_calibration.pdf")
plt.close()
