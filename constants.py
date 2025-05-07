# Origin of the plot
ORIGIN = (35.770771, -78.674804)

# Earth radius
EARTH_RADIUS_M = 6378137 # [m]

# Path to the CSV of tree locations in lat, lon
TREE_LOCATIONS_PATH = "dataset\\TreeLocations-Manual.csv"

# Path to the data logged while traveling
DATA_LOGGER_PATH = "dataset\\GroundPositionLogger.csv"

# Path to the folder with all the images
IMAGE_FOLDER_PATH = "dataset\\images"

# Boolean to plot points of trees
PLOT = True

# Random
RANDOM = False

# The error from the heading on the GPS
HEADING_ERROR_DEG = 5

# Vertical FOV
H_FOV_DEG = 110 # [degrees]

# GPS Margin of Error
GPS_ERROR_M = 5 # [m]

# Radius of area of interest (AOI)
AOI_RADIUS_M = 50 # [m]

# Angle of area of interest (AOI)
AOI_ANGLE_DEG = (H_FOV_DEG + HEADING_ERROR_DEG) / 2
