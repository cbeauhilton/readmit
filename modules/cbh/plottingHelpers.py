def save_pr_curve(target, classifier, test_labels, predicted_labels, figfolder):
    print(f"Saving data for PR curve for {classifier}...")

    import time
    import os
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
    )
    import json

    timestr = time.strftime("%Y-%m-%d-%H%M_")
    average_precision = average_precision_score(test_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(test_labels, predicted_labels)

    d = {}
    d["target"] = target
    d["classifier"] = classifier
    d["figfolder"] = str(figfolder)
    # d["y_test"] = test_labels
    # d["y_pred"] = predicted_labels
    d["average_precision"] = average_precision
    d["precision"] = precision.tolist()
    d["recall"] = recall.tolist()


    figure_title = f"{figfolder}/{target}_{classifier}_Precision_Recall_curve_AP_{average_precision*100:.0f}_"
    filename = figure_title + timestr + ".json"

    with open(filename, "w") as f:
        json.dump(d, f, indent=4)


def plt_pr_curve(target, classifier, test_labels, predicted_labels, figfolder):
    print("Generating PR curve...")

    import time
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
    )
    from inspect import signature

    timestr = time.strftime("%Y-%m-%d-%H%M_")
    average_precision = average_precision_score(test_labels, predicted_labels)
    precision, recall, _ = precision_recall_curve(test_labels, predicted_labels)

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

    figure_title = f"{target}_{classifier}_Precision_Recall_curve_AP_{average_precision*100:.0f}_"
    filename = figure_title + timestr + ".pdf"

    plt.tight_layout()

    plt.savefig(
        (figfolder / filename),
        dpi=1200,
        transparent=False,
        bbox_inches="tight",
    )

    plt.close()

