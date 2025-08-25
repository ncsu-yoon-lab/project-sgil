# Origin of the plot
ORIGIN = (35.770771, -78.674804)

# Earth radius
EARTH_RADIUS_M = 6378137  # [m]

# Path to the CSV of tree locations in lat, lon
TREE_LOCATIONS_PATH = "../dataset/TreeLocations-Manual.csv"

# Path to the data logged while traveling
DATA_LOGGER_PATH = "../dataset/GroundPositionLogger.csv"
# DATA_LOGGER_PATH = "dataset\\position_data_logger-QLoc.csv"
# Path to the folder with all the images
# IMAGE_FOLDER_PATH = "dataset/images"
IMAGE_FOLDER_PATH = "../dataset/images"

# Boolean to plot points of trees
PLOT = True
# PLOT = False

# Random
RANDOM = False

# The error from the heading on the GPS
HEADING_ERROR_DEG = 5

# Vertical and Horizontal FOV
H_FOV_DEG = 110  # [degrees]
V_FOV_DEG = 70  # [degrees]

# Camera specs
CAMERA_HEIGHT_M = 2.57  # [m]
IMAGE_SHAPE = (720, 1280, 3)

# Radius of area of interest (AOI)
AOI_RADIUS_M = 50  # [m]

# Angle of area of interest (AOI)
AOI_ANGLE_DEG = (H_FOV_DEG + HEADING_ERROR_DEG) / 2

# Heuristic score weights (sum does not need to be 1.0)
SCORE_WEIGHT_RMS = 0.45
SCORE_WEIGHT_COMBO_SIZE = 0.25
SCORE_WEIGHT_DISTANCE = 0.15
SCORE_WEIGHT_ANGLE_SPREAD = 0.15

# Sensitivity scales (set based on your units/environment)
SCORE_RMS_SCALE = 1.0  # meters; smaller => stricter on residuals
SCORE_DISTANCE_SCALE = 5.0  # meters; smaller => penalize distance more
