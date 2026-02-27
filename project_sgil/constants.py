from project_sgil.data_structs import Point

IMAGE_TOP_LEFT = Point(35.776006, -78.644325)
IMAGE_BOTTOM_RIGHT = Point(35.772600, -78.637597)

# Origin of the plot
ORIGIN = ((IMAGE_TOP_LEFT.x + IMAGE_BOTTOM_RIGHT.x) / 2.0, (IMAGE_TOP_LEFT.y + IMAGE_BOTTOM_RIGHT.y) / 2.0)

# Earth radius
EARTH_RADIUS_M = 6378137  # [m]

# Path to the CSV of tree locations in lat, lon
TREE_LOCATIONS_PATH = "dataset/tables/RaleighSatellite_manual_trees.csv"

# Path to the data logged while traveling
DATA_LOGGER_PATH = "dataset/tables/results_with_headings.json"
# DATA_LOGGER_PATH = "../dataset/tables/Calibration.csv"
# Path to the folder with all the images
IMAGE_FOLDER_PATH = "dataset/raleigh_images"
# IMAGE_FOLDER_PATH = "../dataset/images-calibration"

OUTPUT_CSV = "annotated_image_data.csv"

# Boolean to plot points of trees
# PLOT = True
PLOT = False

# Boolean to plot wedge debug visuals
PLOT_WEDGES = False #True

# If True, allow wedge debug plots even when not all wedges are matched
PLOT_WEDGES_PARTIAL = False # True

# Random
RANDOM = False

# The error from the heading on the GPS
HEADING_ERROR_DEG = 6

# Heading sweep: try multiple candidate headings around the given yaw
HEADING_SWEEP_ENABLED = True
HEADING_SWEEP_RANGE_DEG = 5  # search yaw ± this many degrees
HEADING_SWEEP_STEP_DEG = 1  # step size in degrees

# Vertical and Horizontal FOV
H_FOV_DEG = 110  # [degrees]
V_FOV_DEG = 70  # [degrees]

# Camera specs
IMAGE_SHAPE = (1242, 2208, 3)

# Radius of area of interest (AOI)
AOI_RADIUS_M = 50  # [m]

# Angle of area of interest (AOI)
AOI_ANGLE_DEG = (H_FOV_DEG + HEADING_ERROR_DEG) / 2

SCORE_WEIGHT_OCCLUSION = 0.8
SCORE_WEIGHT_RMS = 0.6
SCORE_WEIGHT_THETA_MATCH = 0.2
NUMBER_SELECTED_WEIGHT = 0.8

SCORE_RMS_SCALE = 1.0  # meters
TREE_RADIUS_M = 1.1  # meters
THETA_MATCHING_TOLERANCE = 0.8  # degrees

# --- Automated SGIL (results.json) filtering ---
# Process only frames whose numeric index is in [start, end] with a fixed step.
AUTOMATED_FRAME_START = 140
AUTOMATED_FRAME_END = 1000

# Minimum segmentation confidence for using a tree centroid
AUTOMATED_MIN_SEGMENT_CONFIDENCE = 0.0

# If True, ignore frames with no segmentations (otherwise process with no trees)
AUTOMATED_SKIP_IF_NO_TREES = True

# If True, AutomatedSGIL uses the RTK pose (x,y,yaw) for each frame as the
# pose passed into TreeMatcher (no dead-reckoning / carryover).
# If False, it uses the previous estimated pose + RTK delta XY (current behavior).
AUTOMATED_USE_RTK_POSE_EACH_FRAME = False #True

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
HEADING_SCORE_W_NUM_WEDGES = 10.0
HEADING_SCORE_W_THETA_ERROR = 3.0
