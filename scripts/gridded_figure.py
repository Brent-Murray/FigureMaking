import matplotlib.pyplot as plt
from matplotlib import image

def gridded_figure(nrow, ncol, img_list, titles=None):
    """
    Create a gridded figure layout displaying images from a list.

    Parameters:
    nrow (int): Number of rows in the grid.
    ncol (int): Number of columns in the grid.
    img_list (list of str): List of image file paths to be displayed.
    titles (list of str, optional): List of titles for each image. Defaults to None.

    Returns:
    None: This function does not return anything but displays a gridded figure.
    """

    # Set global figure size and axes line width for the plot
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.rcParams["axes.linewidth"] = 1

    # Create subplots with specified number of rows and columns
    fig, axes = plt.subplots(nrow, ncol)
    
    # Flatten the axes array for easy iteration
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        # Load and display each image in the respective subplot
        im = image.imread(img_list[i])
        ax.imshow(im)

        # Add titles to images if provided
        if titles:
            ax.set_title(titles[i], loc="left", fontsize=16)

        # Turn off the axis for each subplot
        ax.axis("off")
        
        
# Runtime
# List of images
images = [r"path/to/img1.png", r"path/to/img2.png", r"path/to/img3.png", r"path/to/img4.png"]

# List of titles
titles = ["Image 1", "Image 2", "Image 3", "Image 4"]

# Make gridded figure
gridded_figure(nrow=2, ncol=2, img_list=images, titles=titles)

# Save figure
plt.savefig(
    r"path/to/output/image.png",
    bbox_inches="tight",
    dpi=600
)