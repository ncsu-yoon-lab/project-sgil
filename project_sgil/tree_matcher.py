from constants import *
import csv
from converter import Converter


class TreeMatcher:

    def __init__(self):
        self.all_sat_tree_locations = []
        self.converter = Converter()

        # Go through the folder of tree locations, convert them to the (x,y) and add them to the list of trees
        with open(TREE_LOCATIONS_PATH, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                point = self.converter.latlon_to_xy((float(row[0]), float(row[1])))
                self.all_sat_tree_locations.append(point)

    def match_trees(self, current_pose, ground_thetas):
        """
        Matches the trees based on the current position (x, y, heading) and the thetas of the trees from the ground view
        Parameters:
            current_pose as the pose estimation as (x, y, heading)
            ground_thetas as the list of thetas of the camera to each tree in the ground view  with left as negative, right as positive
        """

        satellite_trees = self.get_area_of_interest(current_pose)

        wedges = []

        for theta in ground_thetas:
            wedges.append(self.create_wedge(current_pose, satellite_trees, theta))

        self.wedge_matching(wedges)


    def get_area_of_interest(self, current_pose):
        """
        Gets the area of interest of the satellite view
        Parameters:
            current_pose as the pose estimation
            sat_trees as the list of trees inside the area of interest
        """

        pass

    def create_wedge(self, current_pose, satellite_trees, theta):
        """
        Creates a wedge based on current location and theta from the ground view
        Parameters:
            current_pose as the current position and orientation of the ground view
            satellite_trees as the list of trees in the AOI
            theta as the ground view theta to a single tree
        Return:
            wedge formatted as a list of coordinates for trees within that area
        """

        pass

    #TODO Optimize this shit
    def wedge_matching(self, wedges):
        """
        Matches the wedges to the trees to output the location based on its metric of finding the most accurate match
        Parameters:
            wedges as the list of wedges that include the available trees within each wedge
        Return:
            estimated_position as the estimated position of where the vehicle is located
        """

