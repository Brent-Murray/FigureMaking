import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib_scalebar.scalebar import ScaleBar
from tqdm import tqdm


# Read a single-band raster and convert nodata values to NaN
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


# Compute robust min/max values using percentiles for visualization scaling
def normalize(arr):
    vmin = np.nanpercentile(arr, 2)
    vmax = np.nanpercentile(arr, 98)
    return vmin, vmax


# Convert raster values to RGBA using a colormap and transparency for NaNs
def to_rgba(arr, cmap_name, vmin, vmax):
    scaled = (arr - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0, 1)

    rgba = plt.get_cmap(cmap_name)(scaled)

    # Make NaN pixels fully transparent
    rgba[np.isnan(arr), :] = [1, 1, 1, 0]

    return rgba


# Function to truncate colormaps
# Useful for avoiding very dark/light extremes in default matplotlib colormaps
def truncate_colormap(cmap_name, minval=0.2, maxval=1.0, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap_name})", cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


# Update the colorbar dynamically for the currently displayed raster
def update_colorbar(raster_i):

    vmin, vmax = norms[raster_i]

    sm.set_cmap(cmaps[raster_i])
    sm.set_norm(Normalize(vmin=vmin, vmax=vmax))

    cbar.update_normal(sm)

    cbar.ax.set_title(raster_labels[raster_i], fontsize=10, pad=8)

    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")

    cbar.ax.tick_params(labelsize=8)


# Render a single animation frame
def render_frame(frame):
    current_i, next_i, progress, transitioning = frame

    # Static hold frame
    if not transitioning:
        img.set_data(rgba_rasters[current_i])
        update_colorbar(current_i)
        return [img]

    # Compute vertical split position for swipe transition
    split_col = int(width * (1 - progress))

    frame_rgba = rgba_rasters[current_i].copy()

    # Replace right side with incoming raster during transition
    frame_rgba[:, split_col:, :] = rgba_rasters[next_i][:, split_col:, :]

    img.set_data(frame_rgba)

    # During transition, show the legend for the incoming raster
    update_colorbar(next_i)

    return [img]


# Progress bar callback during GIF export
def progress_callback(current_frame, total_frames):
    pbar.update(1)


# Add a custom two-segment alternating black/white scalebar
def add_two_segment_scalebar(
    ax,
    pixel_size=20,
    segment_length_m=500,
    location=(0.08, 0.08),
    height=0.012,
    linewidth=1.2,
    fontsize=10,
    label_in_km=True,
):
    total_length_m = segment_length_m * 2
    total_length_px = total_length_m / pixel_size

    xlim = ax.get_xlim()
    raster_width_px = abs(xlim[1] - xlim[0])

    # Convert scalebar width into axis-relative coordinates
    bar_width_axes = total_length_px / raster_width_px
    bar_height_axes = height

    x0, y0 = location
    x1 = x0 + bar_width_axes / 2
    x2 = x0 + bar_width_axes

    # Left segment (black)
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            bar_width_axes / 2,
            bar_height_axes,
            transform=ax.transAxes,
            facecolor="black",
            edgecolor="black",
            linewidth=linewidth,
            zorder=10,
        )
    )

    # Right segment (white)
    ax.add_patch(
        plt.Rectangle(
            (x1, y0),
            bar_width_axes / 2,
            bar_height_axes,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=linewidth,
            zorder=10,
        )
    )

    # Format labels either in km or m
    if label_in_km:
        middle_label = f"{segment_length_m / 1000:g}"
        right_label = f"{total_length_m / 1000:g} km"
    else:
        middle_label = f"{segment_length_m:g}"
        right_label = f"{total_length_m:g} m"

    label_y = y0 - 0.012

    # Left label (0)
    ax.text(
        x0,
        label_y,
        "0",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        color="black",
        zorder=11,
    )

    # Middle label
    ax.text(
        x1,
        label_y,
        middle_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        color="black",
        zorder=11,
    )

    # Right label
    ax.text(
        x2,
        label_y,
        right_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        color="black",
        zorder=11,
    )


# Paths to rasters
raster_paths = [
    r"D:\MurrayBrent\ABA layers SPL 2018\RMF_20m_T130cm_AGB_ha.tif",  # Above Ground Biomass
    r"D:\MurrayBrent\ABA layers SPL 2018\RMF_20m_T130cm_ba_ha.tif",  # Basal Area
    r"D:\MurrayBrent\ABA layers SPL 2018\RMF_20m_T130cm_dens.tif",  # Density
    r"D:\MurrayBrent\ABA layers SPL 2018\RMF_20m_T130cm_lor.tif",  # Loreys Height
    # r"D:\MurrayBrent\ABA layers SPL 2018\RMF_20m_T130cm_top_height.tif",  # Top Height
    r"D:\MurrayBrent\ABA layers SPL 2018\RMF_20m_T130cm_qmdbh.tif",  # Quadradic Mean DBH
    r"D:\MurrayBrent\ABA layers SPL 2018\RMF_20m_T130cm_V_ha.tif",  # Volume
    # r"D:\MurrayBrent\ABA layers SPL 2018\RMF_20m_T130cm_Vmerch_ha.tif",  # Merchantable Volume
]

# Labels shown above the colorbar for each raster
raster_labels = [
    "Above Ground Biomass (Mg ha$^{-1}$)",
    "Basal Area (m$^2$ ha$^{-1}$)",
    "Density (stems ha$^{-1}$)",
    "Lorey's Height (m)",
    # "Top Height (m)",
    "Quadratic Mean Diameter (cm)",
    "Volume (m$^3$ ha$^{-1}$)",
    # "Merchantable Volume (m$^3$ ha$^{-1}$)",
]

# Colormap configuration for each raster layer
cmaps = [
    truncate_colormap("viridis", 0.15),
    truncate_colormap("inferno", 0.15),
    truncate_colormap("PuBuGn", 0.20),
    truncate_colormap("gist_earth", 0.15),
    # truncate_colormap("plasma", 0.15),
    truncate_colormap("YlGnBu", 0.20),
    truncate_colormap("cividis", 0.15),
    # truncate_colormap("magma", 0.15),
    # truncate_colormap("GnBu", 0.20),
]

# Animation settings
hold_seconds = 1.0
transition_seconds = 0.8
fps = 20
output_path = r"D:\MurrayBrent\Sync\Animation.gif"
figsize = (10, 10)
dpi = 200

# Load all rasters into memory
rasters = [read_raster(p) for p in raster_paths]

# Ensure all rasters share identical dimensions
shapes = [r.shape for r in rasters]
if len(set(shapes)) != 1:
    raise ValueError(f"All rasters must have the same shape. Found: {shapes}")

height, width = rasters[0].shape

# Compute normalization ranges for each raster independently
norms = [normalize(arr) for arr in rasters]

# Precompute RGBA arrays for faster animation rendering
rgba_rasters = [
    to_rgba(arr, cmap, vmin, vmax)
    for arr, cmap, (vmin, vmax) in zip(rasters, cmaps, norms)
]

frames_per_hold = int(hold_seconds * fps)
frames_per_transition = int(transition_seconds * fps)

frame_sequence = []

# Build frame instructions for animation
for i in range(len(rasters)):
    next_i = (i + 1) % len(rasters)

    # Static hold frames
    for _ in range(frames_per_hold):
        frame_sequence.append((i, next_i, 0.0, False))

    # Transition frames
    for f in range(frames_per_transition):
        progress = (f + 1) / frames_per_transition
        frame_sequence.append((i, next_i, progress, True))


fig, ax = plt.subplots(figsize=figsize)
ax.axis("off")

# Initialize displayed raster
img = ax.imshow(rgba_rasters[0], interpolation="none")

# Add custom scalebar
add_two_segment_scalebar(
    ax, pixel_size=20, segment_length_m=10000, label_in_km=True, location=(0.06, 0.02)
)

# Horizontal legend at bottom-right
cax = fig.add_axes([0.58, 0.12, 0.25, 0.02])
# [left, bottom, width, height]

initial_norm = Normalize(vmin=norms[0][0], vmax=norms[0][1])

# ScalarMappable drives the colorbar independently of plotted data
sm = ScalarMappable(norm=initial_norm, cmap=cmaps[0])

sm.set_array([])

# Create horizontal colorbar
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

# Tick marks and labels on bottom
cbar.ax.xaxis.set_ticks_position("bottom")
cbar.ax.xaxis.set_label_position("bottom")

# Title on top
cbar.ax.set_title(raster_labels[0], fontsize=12, pad=8)

cbar.ax.tick_params(labelsize=10)

# Create animation object
ani = animation.FuncAnimation(
    fig, render_frame, frames=frame_sequence, interval=1000 / fps, blit=False
)

# Progress bar for GIF export
pbar = tqdm(total=len(frame_sequence), desc="Rendering GIF")

writer = animation.PillowWriter(fps=fps)

# Save GIF animation
ani.save(output_path, writer=writer, dpi=dpi, progress_callback=progress_callback)

pbar.close()
plt.close(fig)

print(f"Saved GIF to: {output_path}")