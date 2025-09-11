# Origin of the plot
ORIGIN = (35.770771, -78.674804)

# Earth radius
EARTH_RADIUS_M = 6378137  # [m]

# Path to the CSV of tree locations in lat, lon
TREE_LOCATIONS_PATH = "../dataset/tables/TreeLocations-Manual.csv"

# Path to the data logged while traveling
DATA_LOGGER_PATH = "../dataset/tables/GroundPositionLogger.csv"
# DATA_LOGGER_PATH = "../dataset/tables/Calibration.csv"
# Path to the folder with all the images
IMAGE_FOLDER_PATH = "../dataset/images"
# IMAGE_FOLDER_PATH = "../dataset/images-calibration"

OUTPUT_CSV = "annotated_image_data.csv"

# Boolean to plot points of trees
# PLOT = True
PLOT = False

# Random
RANDOM = False

# The error from the heading on the GPS
HEADING_ERROR_DEG = 6

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

SCORE_WEIGHT_OCCLUSION = 0.6
SCORE_WEIGHT_RMS = 0.8
SCORE_WEIGHT_THETA_MATCH = 0.0
NUMBER_SELECTED_WEIGHT = 0.6

SCORE_RMS_SCALE = 1.0  # meters
TREE_RADIUS_M = 1.0  # meters
THETA_MATCHING_TOLERANCE = 1.5  # degrees
