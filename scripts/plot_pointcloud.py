import laspy
import numpy as np
import matplotlib.pyplot as plt


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    Reads a LAS file and extracts point cloud data.

    Parameters:
    pointcloudfile (str): The filepath of the LAS file to be read.
    get_attributes (bool, optional): Flag to indicate whether to return additional point attributes. Defaults to False.
    useevery (int, optional): Step size for sampling the points. Defaults to 1, meaning every point is used.

    Returns:
    np.ndarray: A numpy array of coordinates (x, y, z) from the LAS file.
    dict: (optional) A dictionary of additional attributes if get_attributes is True.
    """

    # Read the LAS file using laspy
    inFile = laspy.read(pointcloudfile)

    # Extract coordinates (X, Y, Z) and downsample if necessary
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]  # Downsample based on 'useevery' parameter

    # If only coordinates are required, return them
    if not get_attributes:
        return coords

    # If additional attributes are requested
    else:
        # Extract all field names from the LAS file
        las_fields = [info.name for info in inFile.points.point_format.dimensions]

        # Store attributes in a dictionary
        attributes = {}
        for las_field in las_fields:  # Include all fields, not just beyond the XYZ
            attributes[las_field] = inFile.points[las_field][::useevery]  # Downsample attribute data

        # Return both coordinates and attributes
        return (coords, attributes)


def plot_pointcloud(coords):
    """
    Plot a 3D point cloud from a set of coordinates.

    This function takes a 3D numpy array 'coords' where each row represents a point in 3D space (x, y, z).
    It plots these points in a 3D scatter plot using Matplotlib.

    Parameters:
    coords (numpy.ndarray): A 2D numpy array of shape (n_points, 3) representing the coordinates of the points in the point cloud.

    Returns:
    None: This function does not return any value. It displays a 3D scatter plot of the point cloud.
    """

    # Setting figure properties
    plt.rcParams["figure.figsize"] = [7.5, 7.5]  # Set figure size
    plt.rcParams["figure.autolayout"] = True      # Enable automatic layout

    # Creating a figure and 3D axis
    fig = plt.figure(linewidth=5)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()  # Hide the axis for a cleaner look

    # Extracting individual coordinates
    x = coords[:, 0]  # Extract x coordinates
    y = coords[:, 1]  # Extract y coordinates
    z = coords[:, 2]  # Extract z coordinates

    # Scatter plot of points
    ax.scatter(x, y, z, s=1, c="#5A5A5A", alpha=1)  # Plot points in 3D space

    # Setting axis limits based on the coordinate ranges
    ax.set_xlim([np.min(x), np.max(x)])  # Set x-axis limits
    ax.set_ylim([np.min(y), np.max(y)])  # Set y-axis limits
    ax.set_zlim([np.min(z), np.max(z)])  # Set z-axis limits
    
    
# Runtime
# Read point cloud
coords = read_las(r"path/to/pointcloud.las")

# Plot pointcloud
plot_pointcloud(coords)

# Save figure
plt.savefig(
    r"path/to/output/image.png",
    bbox_inches="tight",
    dpi=600
)