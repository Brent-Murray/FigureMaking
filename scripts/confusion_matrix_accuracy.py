import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def make_confusion_matrix_with_accuracies(
    cm,
    labels,
    user_accuracy,
    producer_accuracy,
    ratio=[8, 1],
    normalize=False,
    accuracy=None,
    precision=None,
    recall=None,
    f1=None,
    figsize=None,
    stats_text=False,
    font_scale=1.5,
):
    """
    Creates a confusion matrix with additional heatmaps for user and producer accuracies.

    Parameters:
    cm (array-like): Confusion matrix.
    labels (list): List of labels for the matrix axes.
    user_accuracy (array-like): User accuracy values to be displayed in a heatmap.
    producer_accuracy (array-like): Producer accuracy values to be displayed in a heatmap.
    ratio (list, optional): Width and height ratios for the heatmaps. Defaults to [8, 1].
    normalize (bool, optional): Normalize the confusion matrix. Defaults to False.
    accuracy (float, optional): Overall accuracy to display if `stats_text` is True.
    precision (float, optional): Precision value to display if `stats_text` is True.
    recall (float, optional): Recall value to display if `stats_text` is True.
    f1 (float, optional): F1 score to display if `stats_text` is True.
    figsize (tuple, optional): Figure size. Defaults to matplotlib's default figure size.
    stats_text (bool, optional): Whether to display statistics text. Defaults to False.
    font_scale (float, optional): Font scale for the heatmap labels. Defaults to 1.5.

    Returns:
    None: Displays the confusion matrix with heatmaps.
    """
    # Normalization and formatting for the confusion matrix
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = ""

    # Determine the figure size
    if figsize is None:
        figsize = plt.rcParams.get("figure.figsize")

    plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale)  # Setting font scale

    # Determine minimum and maximum values for color scaling
    vmin = np.min(cm)
    vmax = np.max(cm)

    # Creating masks for different heatmap sections
    off_diag_mask = np.eye(*cm.shape, dtype=bool)

    # Custom color palettes
    light_red, dark_red = (1, 0.925, 0.894), (0.4039, 0, 0.051)
    cmap_reds = sns.color_palette(sns.blend_palette([light_red, dark_red], n_colors=255), n_colors=255)
    light_blue, dark_blue = (0.678, 0.847, 0.902), (0.098, 0.098, 0.439)
    cmap_blues = sns.color_palette(sns.blend_palette([light_blue, dark_blue], n_colors=255), n_colors=255)

    # Creating subplots with custom layout
    gs = plt.GridSpec(2, 2, width_ratios=ratio, height_ratios=ratio, hspace=0.1, wspace=0.1)

    # Main confusion matrix heatmap
    ax1 = plt.subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt=fmt, mask=~off_diag_mask, cmap=cmap_blues, cbar=False, xticklabels=labels, yticklabels=labels, ax=ax1)
    sns.heatmap(cm, annot=True, fmt=fmt, mask=off_diag_mask, cmap=cmap_reds, cbar=False, xticklabels=labels, yticklabels=labels, ax=ax1)

    # User accuracy heatmap
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)
    sns.heatmap(user_accuracy, annot=True, fmt=".2f", cmap=cmap_blues, cbar=False, xticklabels=labels, yticklabels=["User's\nAccuracy"], ax=ax2)

    # Producer accuracy heatmap
    ax3 = plt.subplot(gs[0, 1], sharey=ax1)
    sns.heatmap(producer_accuracy, annot=True, fmt=".2f", cmap=cmap_blues, cbar=False, xticklabels=["Producer's\nAccuracy"], yticklabels=labels, ax=ax3)

    # Optional statistics text
    if stats_text:
        stats = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(accuracy, precision, recall, f1)
        ax1.set_xlabel("Predicted Label" + stats)
        ax1.set_ylabel("True Label")
    else:
        ax1.set_xlabel("Predicted Label")
        ax1.set_ylabel("True Label")

    # Adjusting tick labels
    ax1.set_xticklabels(labels, rotation=0)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.yaxis.tick_left()
    ax1.set_yticklabels(labels, rotation=90)
    ax2.xaxis.set_visible(False)
    ax2.set_yticklabels(["User's  \nAccuracy"], rotation=0)
    ax2.yaxis.tick_left()
    ax3.yaxis.set_visible(False)
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position("top")
    
    
# Runtime
# Read in CSV with true and predicted labels
df = pd.read_csv(r"path/to/csv.csv")

# List of class labels
labels = ["Class1", "Class2", "Class3", "Class4"]

# Generate confusion matrix
cm = confusion_matrix(y_true=df["y_true"], y_pred=df["y_pred"])

# Calculate users accuracy for each class
user_accuracies = np.diag(cm) / cm.sum(axis=0)

# Calculate producers accuracy for each class
producer_accuracies = np.diag(cm) / cm.sum(axis=1)

# Calculate accuracy metrix
oa = accuracy_score(y_true=df["y_true"], y_pred=df["y_pred"]) # overall accuracy
f1 = f1_score(y_true=df["y_true"], y_pred=df["y_pred"], average="weighted") # f1 score
recall = recall_score(y_true=df["y_true"], y_pred=df["y_pred"], average="weighted") # recall
precision = precision_score(
    y_true=df["y_true"], y_pred=df["y_pred"], average="weighted"
) # precision

# Make confusion matrix figure
make_confusion_matrix_with_accuracies(
    cm, labels, normalize=False, accuracy=oa, precision=precision, recall=recall, f1=f1, user_accuracy=user_accuracies.reshape(1, -1), producer_accuracy=producer_accuracies.reshape(-1, 1)
)

# Save figure
plt.savefig(
    r"path/to/output/image.png",
    bbox_inches="tight",
    dpi=600
)