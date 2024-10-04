import cv2 as cv

TEST = True

# Image name, correct_lat, correct_lon, correct_heading, zed_x, zed_y, zed_z, estimated_lat, estimated_lon, estimated_heading
# 10:43:13.jpg, 35.77095751, -78.6751402, 169.4002656, -93.3430274, -70.30524988, 3.513918915, 35.77093917, -78.6750895, 191.4499969
TEST_VEHICLE_IMAGE = "test\\10_43_13.jpg"
TEST_SATELLITE_IMAGE = "test\\satellite_image.png"

class Vision:
    def __init__(self):
        # Set up camera
        self.points = []

    def get_image_vehicle(self):
        """
        Gets an image from the camera (This will be replaces with a ROS2 subscription to Zed)
        """

        if TEST:
            image = cv.imread(TEST_VEHICLE_IMAGE)
        else:
            image = None

        return image
    
    def get_image_sat(self, gps_coord):
        """
        Gets the satellite image from the stored satellite tiles
        """

        if TEST:
            gps_coord = (35.77093917, -78.6750895)
            image = cv.imread(TEST_SATELLITE_IMAGE)
        else:
            image = None
        
        return image

    def scan_image_sat(self, image):
        """
        Uses a model to detect the trees in an image and returns the locations of the trees
        """

        if TEST:
            coordinates = [[35.770817, -78.675112],[35.770837, -78.675185]]
            coordinates = [[35.770733, -78.675148], [35.770818, -78.675112], [35.770839, -78.675184],[35.770730, -78.675269]]
        else:
            coordinates = None

        # Returns 2D array of x,y coordinate of all trees
        return coordinates

    def get_thetas(self, locations):
        """
        Converts all the points in the list of locations to a theta
        """

        image_width = 1280

        thetas = []

        for point in locations:
            x_centroid = point[0]

            # Finds the center of the image
            center = image_width / 2

            # Gets the x position of the building based on the center x pixel in the image where the building was identified
            building_x_position = x_centroid - center

            # Field of view horizontal is 110 degrees
            FOV = 110

            # Degrees per pixel
            dpp = FOV / image_width

            # Gets the angle from the lense to the building based on its x position and the degrees per pixel
            theta = dpp * building_x_position

            thetas.append(theta)

        return thetas
    
    def select_trees(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            self.points.append((x,y))

            cv.circle(self.image, (x, y), 5, (0, 0, 255), -1)
            cv.imshow('Image', self.image)

    def scan_image_vehicle(self, image):
        """
        Uses a model to detect the trees in a satellite tile image and returns the
        """

        self.image = image

        if TEST:

            cv.imshow('Image', self.image)
            cv.setMouseCallback('Image', self.select_trees)

            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            tree_centroids = None
        
        return self.points