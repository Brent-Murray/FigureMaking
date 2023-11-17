import os

import matplotlib.pyplot as plt
import pandas as pd

def loss_figure(csv_path, x_col, train_col, val_col, min_max="min"):
    """
    Generates a plot of training and validation loss from a CSV file.

    This function reads loss data from a CSV file and plots the training and validation loss
    over epochs. It also highlights the epoch with the minimum or maximum validation loss, based on 
    the 'min_max' parameter.

    Parameters:
    csv_path (str): Path to the CSV file containing loss data.
    x_col (str): Column name in the CSV for epochs.
    train_col (str): Column name for training loss.
    val_col (str): Column name for validation loss.
    min_max (str, optional): Determines whether to highlight the minimum or maximum validation loss. 
                             Defaults to 'min'. Acceptable values are 'min' or 'max'.

    Raises:
    Exception: If 'min_max' is not 'min' or 'max'.
    """

    # Setting plot aesthetics
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["axes.linewidth"] = 1

    # Reading loss data from the CSV file
    df = pd.read_csv(csv_path)

    # Determining the epoch with best (min/max) validation loss
    if min_max == "min":
        best_loss_epoch = df[x_col].iloc[df[val_col].idxmin()] - 1
        best_val_loss = df[val_col].iloc[df[val_col].idxmin()]
        label = "Lowest Validation Loss"
    elif min_max == "max":
        best_loss_epoch = df[x_col].iloc[df[val_col].idxmax()] - 1
        best_val_loss = df[val_col].iloc[df[val_col].idxmax()]
        label = "Highest Validation Loss"
    else:
        raise Exception("min_max must be either 'min' or 'max'")

    # Plotting training and validation loss
    plt.plot(df[train_col], label="Training loss", color="blue")
    plt.plot(df[val_col], label="Validation loss", color="orange")

    # Highlighting the best epoch
    plt.axvline(x=best_loss_epoch, label=label, color="red", ls=":")
    plt.axhline(y=best_val_loss, color="red", ls=":")
    
    # Position the text with an offset and new line
    offset = 0.01
    if min_max =='min':
        text_x = best_loss_epoch + (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]) * offset
        text_y = best_val_loss + (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) * offset
        text = f'Epoch: {best_loss_epoch}\nLoss: {best_val_loss:.4f}'
        plt.text(text_x, text_y, text, color='r', ha='left', va='bottom')
    elif min_max == 'max':
        text_x = best_loss_epoch + (plt.gca().get_xlim()[1] - plt.gca().get_xlim()[0]) * offset
        text_y = best_val_loss + (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) * -offset
        text = f'Epoch: {best_loss_epoch}\nLoss: {best_val_loss:.4f}'
        plt.text(text_x, text_y, text, color='r', ha='left', va='top')
    else:
        raise Exception("min_max must be either 'min' or 'max'")
        

    # Adding axis labels and legend
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Saving the plot to the specified output path
    plt.savefig(out_path)
    
    
# Runtime
# Create loss figure
loss_figure(
    csv_path=r"path/to/loss.csv",
    x_col="epoch",
    train_col="train_loss",
    val_col="val_loss",
    min_max="min"
)

# Save figure
plt.savefig(
    r"path/to/output/image.png",
    bbox_inches="tight",
    dpi=600
)