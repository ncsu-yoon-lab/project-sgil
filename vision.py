class Vision:
    def __init__(self):
        # Set up camera
        pass

    def get_image(self):
        """
        Gets an image from the camera and returns it
        """
        return None

    def scan_image(self):
        """
        Uses a model to detect the trees in an image and returns the locations of the trees
        """

        # Returns 2D array of x,y coordinate of all trees
        return [[]]

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

    def scan_image_sat(self, gps_coord):
        """
        Uses a model to detect the trees in a satellite tile image and returns the
        """