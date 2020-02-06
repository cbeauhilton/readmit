import json
from inspect import signature
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import interp
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

from cbh import config

targets = [  # "length_of_stay_over_3_days",
    # "length_of_stay_over_5_days",
    "length_of_stay_over_7_days",
    # "length_of_stay_over_14_days",
    # "readmitted3d",
    # "readmitted5d",
    # "readmitted7d",
    # "readmitted30d",
]

target = targets[0]
figfolder = config.FIGURES_DIR/f"{target}/"
json_file = f"{figfolder}/{target}_fig_dict.json"
with open(json_file) as f:
    d = json.load(f)

mean_fpr = np.linspace(0, 1, 100)

for model, data in d.items():
    print(f"Generating ROC curve for {model}...")
    y_test_dict = d[model]["y_test"]
    prob_pos_dict = d[model]["prob_pos"]

    tprs = []
    aucs = []

    plt.figure(figsize=(10,10))
    fig, ax = plt.subplots()

    for i, _ in enumerate(y_test_dict):
        # flatten lists first
        y_test = np.array(list(chain.from_iterable(y_test_dict[f"{i}"])))
        prob_pos = np.array(list(chain.from_iterable(prob_pos_dict[f"{i}"])))


        fpr, tpr, _ = roc_curve(y_test, prob_pos)
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        roc_auc = roc_auc_score(y_test, prob_pos)
        aucs.append(roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=f'Mean ROC (AUC = {mean_auc}:0.2f $\pm$ {std_auc} 0.2f)',
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f"Receiver operating characteristic for {model}")
    ax.legend(loc="lower right")

    plt.savefig(f"{figfolder}/ROC_curve_{target}_{model.replace(' ', '_')}.pdf", bbox_inches="tight", dpi=1200)
    plt.close()


for model, data in d.items():
    print(f"Generating PR curve for {model}...")
    y_test_dict = d[model]["y_test"]
    prob_pos_dict = d[model]["prob_pos"]

    aps = []

    plt.figure(figsize=(10,10))
    fig, ax = plt.subplots()

    for i, _ in enumerate(y_test_dict):
        # flatten lists first
        y_test = np.array(list(chain.from_iterable(y_test_dict[f"{i}"])))
        prob_pos = np.array(list(chain.from_iterable(prob_pos_dict[f"{i}"])))

        precision, recall, _ = precision_recall_curve(y_test, prob_pos)
        aps.append(average_precision_score(y_test, prob_pos))

        step_kwargs = (
            {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}
        )
        ax.step(recall, precision, color="b", alpha=0.3, where="post")
        ax.fill_between(recall, precision, alpha=0.1, color="b", **step_kwargs)

    mean_ap = np.mean(aps)
    std_aps = np.std(aps)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title=f"Precision-Recall for {model}")
    ax.legend([f"Average Precision: {mean_ap:0.2f} $\pm$ {std_aps:0.2f} "], handletextpad=1, handlelength=0, borderpad=1, loc="lower right", framealpha=1)
    plt.savefig(f"{figfolder}/AP_curve_{target}_{model.replace(' ', '_')}.pdf", bbox_inches="tight", dpi=1200)
    plt.close()
