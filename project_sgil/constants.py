from project_sgil.data_structs import Point

IMAGE_TOP_LEFT = Point(35.776006, -78.644325)
IMAGE_BOTTOM_RIGHT = Point(35.772600, -78.637597)

# Origin of the plot
ORIGIN = ((IMAGE_TOP_LEFT.x + IMAGE_BOTTOM_RIGHT.x) / 2.0, (IMAGE_TOP_LEFT.y + IMAGE_BOTTOM_RIGHT.y) / 2.0)

# Earth radius
EARTH_RADIUS_M = 6378137  # [m]

# Path to the CSV of tree locations in lat, lon
TREE_LOCATIONS_PATH = "dataset/tables/RaleighSatellite_manual_trees.csv"
# TREE_LOCATIONS_PATH = "dataset/tables/RaleighSatellite_deepforest_trees_latlon.csv"

# Path to the data logged while traveling
# DATA_LOGGER_PATH = "dataset/tables/results_with_headings.json"
# DATA_LOGGER_PATH = "dataset/tables/manual_json_data.json"
# Path to the folder with all the images
DATA_LOGGER_PATH = "dataset/tables/manual_json_data_w_gaussian.json"
IMAGE_FOLDER_PATH = "dataset/raleigh_images"

# IMAGE_FOLDER_PATH = "../dataset/images-calibration"

OUTPUT_CSV = "annotated_image_data.csv"

SATELLITE_IMAGE_PATH = "dataset/satellite_trees/RaleighSatellite.png"
BACKGROUND_EXTENT = ()

# Boolean to plot points of trees
# PLOT = True
PLOT = False

# Boolean to plot wedge debug visuals
PLOT_WEDGES = False #True

PLOT_THETAS = False

# If True, overlay the satellite image as a faint background on wedge debug plots.
# Kept separate from PLOT_WEDGES so you can generate wedge plots without the heavy background.
PLOT_WEDGES_SAT_BACKGROUND = False

# If True, allow wedge debug plots even when not all wedges are matched
PLOT_WEDGES_PARTIAL = True # True

PLOT_RANGE = (300, 1000)

# Random
RANDOM = False

# The error from the heading on the GPS
HEADING_ERROR_DEG = 3

# JSON field name to read the heading from for tree matching.
# Common options: "rtk_heading", "gps_heading", "rtk_heading_filtered",
#                 "gps_heading_filtered", "noisy_1_heading", etc.
HEADING_JSON_FIELD = "rtk_heading_filtered"

# Heading sweep: try multiple candidate headings around the given yaw
HEADING_SWEEP_ENABLED = False
HEADING_SWEEP_RANGE_DEG = 2  # search yaw ± this many degrees
HEADING_SWEEP_STEP_DEG = 1  # step size in degrees

POSITION_SWEEP_RANGE = 30
POSITION_SWEEP_STEP_SIZE = 1

# Hard gate: reject any pose estimate too far from the snapped/current pose.
# (Prevents weird intersections from dominating scoring.)
MAX_ESTIMATE_DIST_FROM_SNAPPED_M = 30

# Vertical and Horizontal FOV
H_FOV_DEG = 110  # [degrees]
V_FOV_DEG = 70  # [degrees]

# Camera specs
IMAGE_SHAPE = (1242, 2208, 3)

# Radius of area of interest (AOI)
AOI_RADIUS_M = 75  # [m]

# Angle of area of interest (AOI)
AOI_ANGLE_DEG = (H_FOV_DEG + HEADING_ERROR_DEG) / 2

# Min change in dx and dy for the gps
MIN_DX = MIN_DY = 1.0

SCORE_WEIGHT_OCCLUSION = 0.8
SCORE_WEIGHT_RMS = 0.6
SCORE_WEIGHT_THETA_MATCH = 0.2
NUMBER_SELECTED_WEIGHT = 0.8

SCORE_RMS_SCALE = 1.0  # meters
TREE_RADIUS_M = 1.1  # meters
THETA_MATCHING_TOLERANCE = 0.8  # degrees

# --- Automated SGIL image skipping ---
# List of (start_inclusive, end_exclusive) frame index ranges to skip.
# Example: IMAGES_TO_SKIP = [(200, 250), (1200, 1400)]
# IMAGES_TO_SKIP: list[tuple[int, int]] = [(318, 605), (786, 3633), (5036, 6500)]
IMAGES_TO_SKIP: list[tuple[int, int]] = [(318, 605), (786, 1790), (5036, 6500)]

# If True, skipped frames still appear in the results table with their
# GPS-delta-propagated position (no tree matching is performed).
INCLUDE_SKIPPED_FRAMES = True

# --- Automated SGIL (results.json) filtering ---
# Process only frames whose numeric index is in [start, end] with a fixed step.
AUTOMATED_FRAME_START = 140 # 288
AUTOMATED_FRAME_END = 2047 # 771 # 4000

# Minimum segmentation confidence for using a tree centroid
AUTOMATED_MIN_SEGMENT_CONFIDENCE = 0.0

# If True, ignore frames with no segmentations (otherwise process with no trees)
AUTOMATED_SKIP_IF_NO_TREES = True

# If True, AutomatedSGIL uses the RTK pose (x,y,yaw) for each frame as the
# pose passed into TreeMatcher (no dead-reckoning / carryover).
# If False, it uses the previous estimated pose + RTK delta XY (current behavior).
AUTOMATED_USE_RTK_POSE_EACH_FRAME = False

# World frame:
#   +x = East, +y = North   (see Converter.latlon_to_xy)
# Yaw:
#   yaw_deg = 0 points along +x (East)
#   yaw_deg = +90 points along +y (North)
#   (see Converter.heading_to_yaw and DebugVisualizer heading arrow)
# Robot/body frame used for sensor extrinsics in this project:
#   +x_body = forward
#   +y_body = left

RTK_TO_CAMERA_OFFSET_X_FWD_M = -0.53
RTK_TO_CAMERA_OFFSET_Y_LEFT_M = -(1.06 / 2.0 - 0.06)

# Backwards-compatible aliases (deprecated)
RTK_TO_CAMERA_OFFSET_X_M = RTK_TO_CAMERA_OFFSET_X_FWD_M
RTK_TO_CAMERA_OFFSET_Y_M = RTK_TO_CAMERA_OFFSET_Y_LEFT_M

# --- Heading sweep ranking (post pose-score heuristic) ---
# Used by TreeMatcher._calculate_heading_score().
# These are the ONLY tuning knobs for heading sweep selection.
HEADING_SCORE_W_DELTA_YAW = 1.0
HEADING_SCORE_W_NUM_WEDGES = 100.0
HEADING_SCORE_W_THETA_ERROR = 30.0

# If set, only generate wedge debug plots for this specific frame index.
# Set to None to allow wedge plots for all frames.
WEDGE_PLOT_ONLY_IMAGE = 605

# If True, only generate the wedge plot for the final chosen combo (best estimate).
# If False, TreeMatcher may generate many 'combo_*.png' plots for each candidate combination.
PLOT_ONLY_CHOSEN_COMBO = True

# --- Wedge plot performance knobs ---
# These debug plots can be expensive. Turn these on to speed them up.
WEDGE_PLOT_LABEL_TREES = False  # if True, annotate each tree with lat/lon text
WEDGE_PLOT_DPI = 110  # lower DPI saves faster; 100-120 is usually plenty
WEDGE_PLOT_TIGHT_BBOX = False  # bbox_inches='tight' is slow; False is faster

# More wedge plot performance knobs
WEDGE_PLOT_SHOW_CANDIDATE_TREES = False  # per-wedge candidate scatters are expensive
WEDGE_PLOT_SHOW_RAYS = True  # draw matched-tree rays (can disable for speed)
WEDGE_PLOT_SHOW_LEGEND = False  # legend layout can be slow
WEDGE_PLOT_SHOW_GRID = False
DEBUG_WEDGE_COMBOS_MAX_PRINT = 25
DEBUG_WEDGE_COMBOS_FILTER_TREE_IDS: list[int] | None = None

# Maximum number of wedge→tree combinations to evaluate per heading candidate.
# Prevents combinatorial explosion when many wedges / candidate trees exist.
# Set to None for unlimited (original behaviour — may hang on dense scenes).
MAX_WEDGE_COMBINATIONS: int | None = 50_000

