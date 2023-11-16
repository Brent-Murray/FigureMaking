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

def make_confusion_matrix(
    cm,
    labels,
    normalize=False,
    accuracy=None,
    precision=None,
    recall=None,
    f1=None,
    figsize=None,
):
    """
    Generates and displays a confusion matrix heatmap.

    Parameters:
    - cm (array-like): Confusion matrix to be visualized.
    - labels (list): List of label names corresponding to the classes.
    - normalize (bool, optional): If True, normalizes the confusion matrix.
    - accuracy (float, optional): Accuracy score to be displayed.
    - precision (float, optional): Precision score to be displayed.
    - recall (float, optional): Recall score to be displayed.
    - f1 (float, optional): F1 score to be displayed.
    - figsize (tuple, optional): Size of the figure. Defaults to matplotlib's default figure size.

    Returns:
    - None: The function directly visualizes the heatmap without returning any value.
    """
    # Setting the default figure size
    plt.rcParams["figure.figsize"] = (15, 15)
    
    # Normalizing the confusion matrix if required
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"  # Format for normalized values
    else:
        fmt = ""  # No special formatting for non-normalized values

    # Set the figure size
    if figsize == None:
        figsize = plt.rcParams.get("figure.figsize")  # Get default figure size if not set

    # Heatmap visualization setup
    plt.figure(figsize=figsize)
    vmin = np.min(cm)  # Minimum value in the confusion matrix for colormap scaling
    vmax = np.max(cm)  # Maximum value in the confusion matrix for colormap scaling
    off_diag_mask = np.eye(*cm.shape, dtype=bool)  # Mask for off-diagonal elements

    # Heatmap for off-diagonal elements
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        mask=~off_diag_mask,
        cmap="Blues",
        cbar=False,
        linewidths=1,
        linecolor="black",
        xticklabels=labels,
        yticklabels=labels,
    )

    # Heatmap for diagonal elements
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        mask=off_diag_mask,
        cmap="Reds",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )

    # Adding performance metrics (if provided) to the plot
    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        accuracy, precision, recall, f1
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label" + stats_text)  # Including the metrics in the x-label

    
# Runtime
# Read in CSV with true and predicted labels
df = pd.read_csv(r"path/to/csv.csv")

# List of class labels
labels = ["Class1", "Class2", "Class3", "Class4"]

# Generate confusion matrix
cm = confusion_matrix(y_true=df["y_true"], y_pred=df["y_pred"])

# Calculate accuracy metrix
oa = accuracy_score(y_true=df["y_true"], y_pred=df["y_pred"]) # overall accuracy
f1 = f1_score(y_true=df["y_true"], y_pred=df["y_pred"], average="weighted") # f1 score
recall = recall_score(y_true=df["y_true"], y_pred=df["y_pred"], average="weighted") # recall
precision = precision_score(
    y_true=df["y_true"], y_pred=df["y_pred"], average="weighted"
) # precision

# Make confusion matrix figure
make_confusion_matrix(
    cm, labels, normalize=False, accuracy=oa, precision=precision, recall=recall, f1=f1
)

# Save figure
plt.savefig(
    r"path/to/output/image.png",
    bbox_inches="tight",
    dpi=600
)