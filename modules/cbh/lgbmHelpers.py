import json
import os
import time
import traceback
from inspect import signature

import h5py
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
try:
    import lightgbm as lgb
except BaseException:
    print("lightgbm not available\n")

import matplotlib
import matplotlib.pylab as pl

# import cbh.texfig first to configure Matplotlib's backend
# import cbh.texfig as texfig
# then, import PyPlot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import cbh.config
from mlxtend.evaluate import bootstrap_point632_score

jsonpickle_numpy.register_handlers()


try:
    import cPickle as pickle
except BaseException:
    import pickle

def bootstrap_estimate_and_ci(estimator, X, y, scoring_func=None, random_seed=42,
                              method='.632', alpha=0.05, n_splits=200):
      scores = bootstrap_point632_score(estimator, X, y, scoring_func=scoring_func,
      n_splits=n_splits, random_seed=random_seed,  method=method)

      ci_dict = {}

      if isinstance(scoring_func, list):
            for func in scoring_func:
                  func_name = func.__name__
                  ci_dict[f"{func_name}"] = {}
                  ci_dict[f"{func_name}"]["estimate"] = np.mean(scores[func_name])
                  ci_dict[f"{func_name}"]["lower_bound"] = np.percentile(scores[func_name], 100*(alpha/2)) 
                  ci_dict[f"{func_name}"]["upper_bound"] = np.percentile(scores[func_name], 100*(1-alpha/2)) 
                  ci_dict[f"{func_name}"]["stderr"] = np.std(scores[func_name]) 

      else:
            estimate = np.mean(scores)
            lower_bound = np.percentile(scores, 100*(alpha/2))
            upper_bound = np.percentile(scores, 100*(1-alpha/2))
            stderr = np.std(scores)

            func_name = scoring_func.__name__
            ci_dict[f"{func_name}"] = {"estimate": estimate, "lower_bound": lower_bound, "upper_bound": upper_bound, "stderr": stderr}
      
      return ci_dict

# def bootstrap_estimate_and_ci(
#     estimator,
#     X,
#     y,
#     scoring_func=None,
#     random_seed=42,
#     method=".632",
#     alpha=0.05,
#     n_splits=200,
# ):
# # https://gist.github.com/roncho12/60178f12ea4c3a74764fd645c6f2fe13 
#     scores = bootstrap_point632_score(
#         estimator,
#         X,
#         y,
#         scoring_func=scoring_func,
#         n_splits=n_splits,
#         random_seed=random_seed,
#         method=method,
#     )
#     estimate = np.mean(scores)
#     lower_bound = np.percentile(scores, 100 * (alpha / 2))
#     upper_bound = np.percentile(scores, 100 * (1 - alpha / 2))
#     stderr = np.std(scores)

#     return estimate, lower_bound, upper_bound, stderr
    
class lgbmClassificationHelpers:
    """ 
    target: the thing you are predicting (y)
    class_threshold: cutoff for predicting class 1 vs 2, etc.
    """

    def __init__(
        self,
        target,
        class_thresh,
        gbm_model,
        evals_result,
        features,
        labels,
        train_features,
        train_labels,
        test_features,
        test_labels,
        valid_features,
        valid_labels,
        figfolder,
        datafolder,
        modelfolder,
        tablefolder,
        calibrate_please=False,
    ):
        self.target = target
        self.class_thresh = class_thresh
        self.gbm_model = gbm_model
        self.evals_result = evals_result
        self.features = features
        self.labels = labels
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.valid_features = valid_features
        self.valid_labels = valid_labels
        self.figfolder = figfolder
        self.datafolder = datafolder
        self.modelfolder = modelfolder
        self.tablefolder = tablefolder
        self.calibrate_please = calibrate_please
        self.n_features = str(len(self.train_features.columns))
        self.timestr = time.strftime("_%Y-%m-%d-%H%M_")
        self.timestr_d = time.strftime("_%Y-%m-%d_")
        file_title = f"{self.target}_{self.n_features}_everything_"
        ext = ".h5"
        title = file_title + self.timestr_d + ext
        self.h5_file = self.modelfolder / title

    def lgbm_save_ttv_split(self):
        n_features = self.n_features
        h5_file = self.h5_file

        # delete the h5 file if it exists
        try:
            os.remove(h5_file)
            print("Removed old file", h5_file)
        except OSError:
            # print(traceback.format_exc())
            print("File did not exist, making new...")
        print("Saving TTV split to .h5 file...")
        self.train_features.to_hdf(
            h5_file, key="train_features", mode="w", format="table"
        )
        # using mode="w" for the first one will overwrite an existing file,
        # and so avoid key conflicts on files produced the same day
        self.train_labels.to_hdf(h5_file, key="train_labels", format="table")
        self.test_features.to_hdf(h5_file, key="test_features", format="table")
        self.test_labels.to_hdf(h5_file, key="test_labels", format="table")
        self.valid_features.to_hdf(h5_file, key="valid_features", format="table")
        self.valid_labels.to_hdf(h5_file, key="valid_labels", format="table")
        self.labels.to_hdf(h5_file, key="labels", format="table")
        self.features.to_hdf(h5_file, key="features", format="table")
        print(f"TTV split available at {h5_file} .")

    def lgbm_save_model_to_pkl_and_h5(self):
        """
        To retrieve pickled model, use something like:
        `pkl_model = metricsgen.save_model_to_pickle()`

        then: 

        `with open(pkl_model, "rb") as fin:
                gbm_model = pickle.load(fin)` 
        """

        print("Dumping model with pickle...")
        n_features = self.n_features
        file_title = f"{self.target}_{n_features}_features_MODEL_"
        ext = ".pickle"
        title = file_title + n_features + self.timestr + ext
        pkl_model = self.modelfolder / title
        with open(pkl_model, "wb") as fout:
            pickled = pickle.dump(self.gbm_model, fout)
        print(pkl_model)
        print("JSONpickling the model...")
        frozen = jsonpickle.encode(self.gbm_model)
        print("Saving GBM model to .h5 file...")
        h5_file = self.h5_file
        with h5py.File(h5_file, "a") as f:
            try:
                f.create_dataset("gbm_model", data=frozen)
            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
                try:
                    del f["gbm_model"]
                    f.create_dataset("gbm_model", data=frozen)
                    print("Successfully deleted old gbm model and saved new one!")
                except:
                    print("Old gbm model persists...")
        print(h5_file)
        return pkl_model

    def lgbm_save_feature_importance_plot(self):
        print("Plotting feature importances...")
        n_features = self.n_features
        ax = lgb.plot_importance(
            self.gbm_model, figsize=(5, 20), importance_type="gain", precision=2
        )
        figure_title = f"{self.target}_Feature_Importances_"
        ext = ".png"
        title = figure_title + n_features + self.timestr + ext
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=False, bbox_inches="tight"
        )
        try:
            title1 = figure_title + n_features + self.timestr
            title1 = str(self.figfolder) + "/" + title1
            plt.savefig(title1 + ".pdf", bbox_inches="tight")
            print(title1)
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
        plt.close()



    def lgbm_classification_results(self, bootstrap=False):
        n_features = self.n_features

        # Predict probabilities
        predicted_labels = self.gbm_model.predict_proba(self.test_features)
        predicted_labels = predicted_labels[:, 1]

        # histogram of predicted probabilities

        print("Generating histogram of probabilities...")
        plt.hist(predicted_labels, bins=8)
        plt.xlim(0, 1)  # x-axis limit from 0 to 1
        plt.title("Histogram of predicted probabilities")
        plt.xlabel("Predicted probability")
        plt.ylabel("Frequency")
        figure_title = f"{self.target}_Probability_Histogram_"
        ext = ".png"
        title = figure_title + n_features + self.timestr + ext
        print(title)
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=False, bbox_inches="tight"
        )
        try:
            title1 = figure_title + n_features + self.timestr
            title1 = str(self.figfolder) + "/" + title1
            plt.savefig(title1 + ".pdf", bbox_inches="tight")
            # print(title1)
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
        plt.close()

        fpr, tpr, threshold = metrics.roc_curve(self.test_labels, predicted_labels)
        roc_auc = auc(fpr, tpr)

        print("Generating ROC curve...")
        # plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, "b", label=f"AUC {roc_auc:.2f}")
        plt.legend(handletextpad=0, handlelength=0, loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([-0.011, 1.011])
        plt.ylim([-0.011, 1.011])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        figure_title = (
            f"{self.target}_Receiver_Operating_Characteristic_AUC_{roc_auc*100:.0f}_"
        )
        ext = ".png"
        title = figure_title + n_features + self.timestr + ext
        print(title)
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=False, bbox_inches="tight"
        )
        try:
            title1 = figure_title + n_features + self.timestr
            title1 = str(self.figfolder) + "/" + title1
            plt.savefig(title1 + ".pdf")

        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
        plt.close()
        print(f"{self.target} {n_features} ROC AUC {roc_auc:.2f}")

        print("Generating PR curve...")
        average_precision = average_precision_score(self.test_labels, predicted_labels)
        precision, recall, _ = precision_recall_curve(
            self.test_labels, predicted_labels
        )

        step_kwargs = (
            {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
        )
        # plt.title(f"Precision-Recall Curve")
        plt.step(recall, precision, color="b", alpha=0.2, where="post")
        plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(
            [f"Average Precision: {average_precision:0.2f}"],
            handletextpad=0,
            handlelength=0,
            loc="lower right",
        )
        figure_title = (
            f"{self.target}_Precision_Recall_curve_AP_{average_precision*100:.0f}_"
            % average_precision
        )
        ext = ".png"
        title1 = figure_title + n_features + self.timestr + ext
        print(title1)
        plt.savefig(
            (self.figfolder / title1), dpi=1200, transparent=False, bbox_inches="tight"
        )
        try:
            title1 = figure_title + n_features + self.timestr
            title1 = str(self.figfolder) + "/" + title1
            plt.savefig(title1 + ".pdf", bbox_inches="tight")
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
        plt.close()
        print(f"{self.target} Average precision-recall score: {average_precision:.2f}")

        print("Generating calibration curve...")
        # Using validation for predict proba here
        # so the calibration can be on a disjoint dataset
        predicted_labels_cal = self.gbm_model.predict_proba(self.valid_features)
        predicted_labels_cal = predicted_labels_cal[:, 1]
        brier_score = brier_score_loss(self.valid_labels, predicted_labels_cal)
        print("Brier score without optimized calibration: %1.3f" % brier_score)

        gb_y, gb_x = calibration_curve(
            self.valid_labels, predicted_labels_cal, n_bins=50
        )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(gb_x, gb_y, marker=".", color="red")
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.legend(
            [f"Brier Score Loss: {brier_score:.2f}"],
            handletextpad=0,
            handlelength=0,
            loc="lower right",
        )
        figure_title = f"{self.target}_Calibration_curve_{brier_score*100:.0f}_"
        ext = ".png"
        title1 = figure_title + n_features + self.timestr + ext
        print(title1)
        plt.savefig(
            (self.figfolder / title1), dpi=1200, transparent=False, bbox_inches="tight"
        )
        try:
            title1 = figure_title + n_features + self.timestr
            title1 = str(self.figfolder) + "/" + title1
            plt.savefig(title1 + ".pdf", bbox_inches="tight")
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
        plt.close()

        brier_score_cal_sig = []
        brier_score_cal_iso = []

        if self.calibrate_please:

            print("Preparing to calibrate...")
            self.valid_features.columns = (
                self.valid_features.columns.str.strip()
                .str.replace("_", " ")
                .str.replace("__", " ")
                .str.capitalize()
            )

            # Calibration doesn't like "category" dtype, so switch to category codes here.
            # This requires that you already set dtype to "category" for all columns of interest.

            train_features_codes = self.train_features.copy()
            valid_features_codes = self.valid_features.copy()
            test_features_codes = self.test_features.copy()

            for col in list(train_features_codes.select_dtypes(include="category")):
                train_features_codes[col] = train_features_codes[col].cat.codes
            for col in list(valid_features_codes.select_dtypes(include="category")):
                valid_features_codes[col] = valid_features_codes[col].cat.codes
            for col in list(test_features_codes.select_dtypes(include="category")):
                test_features_codes[col] = test_features_codes[col].cat.codes

            print("Calibrating (sigmoid)...")
            cccv_sig = CalibratedClassifierCV(
                self.gbm_model, cv="prefit", method="sigmoid"
            )
            cccv_sig.fit(test_features_codes, self.test_labels)
            predicted_labels_cal_sig = cccv_sig.predict_proba(valid_features_codes)
            predicted_labels_cal_sig = predicted_labels_cal_sig[:, 1]
            brier_score_cal_sig = brier_score_loss(
                self.valid_labels, predicted_labels_cal_sig
            )
            print(
                "Brier score with sigmoid optimized calibration: %1.3f"
                % brier_score_cal_sig
            )

            gb_y, gb_x = calibration_curve(
                self.valid_labels, predicted_labels_cal_sig, n_bins=50
            )
            plt.plot([0, 1], [0, 1], linestyle="--")
            # plot model reliability
            plt.plot(gb_x, gb_y, marker=".", color="red")
            plt.xlabel("Predicted probability")
            plt.ylabel("True probability")
            plt.legend(
                [f"Brier Score Loss: {brier_score:.2f}"],
                handletextpad=0,
                handlelength=0,
                loc="lower right",
            )
            figure_title = f"{self.target}_Calibration_curve_sigmoid_calibration_{brier_score_cal_sig*100:.0f}_"
            ext = ".png"
            title1 = figure_title + n_features + self.timestr + ext
            print(title1)
            plt.savefig(
                (self.figfolder / title1),
                dpi=1200,
                transparent=False,
                bbox_inches="tight",
            )
            plt.close()

            print("Calibrating (isotonic)...")
            cccv_iso = CalibratedClassifierCV(
                self.gbm_model, cv="prefit", method="isotonic"
            )
            cccv_iso.fit(test_features_codes, self.test_labels)
            predicted_labels_cal_iso = cccv_iso.predict_proba(valid_features_codes)
            predicted_labels_cal_iso = predicted_labels_cal_iso[:, 1]
            brier_score_cal_iso = brier_score_loss(
                self.valid_labels, predicted_labels_cal_iso
            )
            print(
                "Brier score with isotonic optimized calibration: %1.3f"
                % brier_score_cal_iso
            )

            gb_y, gb_x = calibration_curve(
                self.valid_labels, predicted_labels_cal_iso, n_bins=50
            )
            plt.plot([0, 1], [0, 1], linestyle="--")
            # plot model reliability
            plt.plot(gb_x, gb_y, marker=".", color="red")
            plt.xlabel("Predicted probability")
            plt.ylabel("True probability")
            plt.legend(
                [f"Brier Score Loss: {brier_score:.2f}"],
                handletextpad=0,
                handlelength=0,
                loc="lower right",
            )
            figure_title = f"{self.target}_Calibration_curve_isotonic_calibration_{brier_score_cal_iso*100:.0f}_"
            ext = ".png"
            title1 = figure_title + n_features + self.timestr + ext
            print(title1)
            plt.savefig(
                (self.figfolder / title1),
                dpi=1200,
                transparent=False,
                bbox_inches="tight",
            )
            plt.close()

        print("Generating classification report...")
        predicted_labels = self.gbm_model.predict(self.test_features)
        # predicted_labels = predicted_labels[:, 1]
        c_range = predicted_labels.size

        # convert into binary values for classification report
        print("Classification threshold is ", self.class_thresh)
        for i in range(0, c_range):
            if (
                predicted_labels[i] >= self.class_thresh
            ):  # set threshold to desired value
                predicted_labels[i] = 1
            else:
                predicted_labels[i] = 0

        accuracy = accuracy_score(self.test_labels, predicted_labels)
        ap_score = average_precision_score(self.test_labels, predicted_labels)
        bsl = brier_score_loss(self.test_labels, predicted_labels)
        f1 = f1_score(self.test_labels, predicted_labels)
        pr_score = precision_score(self.test_labels, predicted_labels)
        re_score = recall_score(self.test_labels, predicted_labels)
        mcc = matthews_corrcoef(self.test_labels, predicted_labels)

        if bootstrap:
            scoring_funcs = [
                accuracy_score,
                brier_score_loss,
                f1_score,
                precision_score,
                recall_score,
                matthews_corrcoef,
            ]
            
            scores = {}
            scores[self.target] = {}
            for scoring_func in scoring_funcs:
                est, low, up, stderr = bootstrap_estimate_and_ci(
                    estimator=self.gbm_model,
                    X=self.features,
                    y=self.labels,
                    method=".632+",
                    n_splits=200,
                    scoring_func=scoring_func,
                )
                scores[self.target][f"{scoring_func.__name__}"] = {
                    "estimate": est,
                    "lower bound": low,
                    "upper bound": up,
                    "standard error": stderr,
                }
                print(scores)

            scoring_funcs = [average_precision_score, roc_auc_score]
            cloned_estimator = clone(self.gbm_model)
            cloned_estimator.predict = cloned_estimator.decision_function

            for scoring_func in scoring_funcs:
                est, low, up, stderr = bootstrap_estimate_and_ci(
                    estimator=cloned_estimator,
                    X=self.features,
                    y=self.labels,
                    method=".632+",
                    n_splits=200,
                    scoring_func=scoring_func,
                )
                scores[self.target][f"{scoring_func.__name__}"] = {
                    "estimate": est,
                    "lower bound": low,
                    "upper bound": up,
                    "standard error": stderr,
                }
                print(scores)

            print(scores)

            with open(cbh.config.SCORES_JSON, "w") as f:
                json.dump(scores, f, indent=4)

        print(f"Accuracy of GBM classifier for {self.target}: ", accuracy)
        # from imblearn.metrics import classification_report_imbalanced
        # print(classification_report_imbalanced(self.test_labels, predicted_labels))

        conf_mx = metrics.confusion_matrix(self.test_labels, predicted_labels)
        fig = sns.heatmap(conf_mx, annot=True, fmt="d", cbar=False, linewidths=0.5)
        fig = fig.get_figure()
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        figure_title = f"{self.target}_Confusion_Matrix_AUC_{roc_auc*100:.0f}_"
        ext = ".png"
        title = figure_title + self.timestr + ext
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=False, bbox_inches="tight"
        )
        try:
            title1 = figure_title + n_features + self.timestr
            title1 = str(self.figfolder) + "/" + title1
            plt.savefig(title1 + ".pdf", bbox_inches="tight")
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
        plt.close()

        conf_mx = pd.DataFrame(conf_mx)
        print(self.test_labels)
        # prevalence = len(self.test_labels[1] == 1) / len(self.test_labels)
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

        # print(self.target)
        n_pts = len(predicted_labels)
        print("N:", n_pts)
        # print("Prevalence:", prevalence)
        print("TPR:", TPR[1])
        print("TNR:", TNR[1])
        print("PPV:", PPV[1])
        print("NPV:", NPV[1])
        print("FPR:", FPR[1])
        print("FNR:", FNR[1])
        print("FDR:", FDR[1])
        print("ACC:", ACC[1])

        d = [
            [
                self.timestr,
                self.target,
                n_features,
                self.class_thresh,
                n_pts,
                # prevalence,
                TPR[1],
                TNR[1],
                PPV[1],
                NPV[1],
                FPR[1],
                FNR[1],
                FDR[1],
                ACC[1],
                TP[1],
                TN[1],
                FP[1],
                FN[1],
                roc_auc,
                average_precision,
                f1,
                mcc,
                brier_score,
                brier_score_cal_sig,
                brier_score_cal_iso,
            ]
        ]

        df = pd.DataFrame(
            d,
            columns=(
                "Time",
                "Target",
                "Number of Features",
                "Threshold",
                "Number of Patients",
                # "Prevalence of Target",
                "TPR_Recall_Sensitivity",
                "TNR_Specificity",
                "PPV_Precision",
                "NPV",
                "FPR",
                "FNR",
                "FDR",
                "ACC",
                "TP",
                "TN",
                "FP",
                "FN",
                "ROC AUC",
                "Average Precision",
                "F1 Score",
                "Matthews Correlation Coefficient",
                "Brier Score Loss",
                "Brier Score Loss Sigmoid Calibration",
                "Brier Score Loss Isotonic Calibration",
            ),
        )

        df.to_csv(
            self.tablefolder / "classifiertrainingreports.csv", mode="a", header=True
        )
        d2 = [
            [
                self.timestr,
                self.target,
                n_features,
                roc_auc,
                average_precision,
                brier_score,
            ]
        ]

        df2 = pd.DataFrame(
            d2,
            columns=(
                "Time",
                "Target",
                "Number of Features",
                "ROC AUC",
                "Average Precision",
                "Brier Score Loss",
            ),
        )
        feature_selection_csv = self.tablefolder / "featureselectiontrainingreports.csv"
        if not os.path.isfile(feature_selection_csv):
            df2.to_csv(feature_selection_csv, mode="a", header=True)
        elif os.path.isfile(feature_selection_csv):
            df2.to_csv(feature_selection_csv, mode="a", header=False)
        ###TODO: save this to h5 file as well

        import cbh.config as config

        figures_path = config.FIGURES_DIR
        feat_sel = pd.read_csv(feature_selection_csv)

        # cf.set_config_file(world_readable=True, theme="pearl", offline=True)
        # init_notebook_mode(connected=True)
        # cf.go_offline()

        # feat_sel.iplot(
        #     kind="bubble",
        #     x="Number of Features",
        #     y="ROC AUC",
        #     size="Brier Score Loss",
        #     text="Target",
        #     xTitle="Number of Features",
        #     yTitle="ROC AUC",
        #     filename=os.path.join(
        #         figures_path, "Training Metrics wrt Number of Features"
        #     ),
        #     asPlot=True,
        # )

        return roc_auc, average_precision, brier_score, n_features


class lgbmRegressionHelpers:
    """ 
    target: the thing you are predicting (y).
    gbm_model: trained model.
    class_threshold: cutoff for predicting class 1 vs 2, etc.
    evals_result: evals_result.
    """

    def __init__(
        self,
        target,
        gbm_model,
        evals_result,
        test_features,
        test_labels,
        valid_features,
        valid_labels,
        figfolder,
        datafolder,
        modelfolder,
        tablefolder,
    ):
        self.target = target
        self.class_thresh = class_thresh
        self.gbm_model = gbm_model
        self.evals_result = evals_result
        self.test_features = test_features
        self.test_labels = test_labels
        self.valid_features = valid_features
        self.valid_labels = valid_labels
        self.figfolder = figfolder
        self.datafolder = datafolder
        self.modelfolder = modelfolder
        self.tablefolder = tablefolder
        self.timestr = time.strftime("%Y-%m-%d-%H%M")
        self.timestr_d = time.strftime("%Y-%m-%d")

    def save_model_to_pickle(self):
        """
        To retrieve pickled model, use something like:
        `pkl_model = metricsgen.save_model_to_pickle()`

        then: 

        `with open(pkl_model, "rb") as fin:
                gbm_model = pickle.load(fin)` 
        """

        print("Dumping model with pickle...")
        file_title = f"{self.target}_MODEL_"
        ext = ".pickle"
        title = file_title + self.timestr_d + ext
        pkl_model = self.modelfolder / title
        with open(pkl_model, "wb") as fout:
            pickle.dump(self.gbm_model, fout)
        print(pkl_model)
        return pkl_model

    def save_feature_importance_plot(self):
        print("Plotting feature importances...")
        ax = lgb.plot_importance(
            self.gbm_model, figsize=(5, 20), importance_type="gain", precision=2
        )
        figure_title = f"{self.target}_Feature_Importances_"
        ext = ".png"
        title = figure_title + self.timestr + ext
        plt.savefig(
            (self.figfolder / title), dpi=1200, transparent=False, bbox_inches="tight"
        )
        try:
            title1 = figure_title + n_features + self.timestr
            title1 = str(self.figfolder) + "/" + title1
            plt.savefig(title1 + ".pdf", bbox_inches="tight")
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
        plt.close()
