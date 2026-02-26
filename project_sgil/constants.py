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
DATA_LOGGER_PATH = "dataset/tables/results.json"
# DATA_LOGGER_PATH = "../dataset/tables/Calibration.csv"
# Path to the folder with all the images
IMAGE_FOLDER_PATH = "dataset/raleigh_images"
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
IMAGE_SHAPE = (720, 1280, 3)

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
