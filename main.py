from vision import Vision
from converter import Converter
import math

def main():
    vehicle_image = vision.get_image_vehicle()

    # Only for testing purposes
    gps_coord = [35.77093917, -78.6750895, 191.4499969]

    # Gets the global heading
    heading = gps_coord[2]

    # Gets the satellite image at the specified coordinate
    satellite_image = vision.get_image_sat(gps_coord)

    # Gets the centroid of the trees detected in the image
    tree_centroids = vision.scan_image_vehicle(vehicle_image)

    # Converts the x of each centroid to a theta from view point to tree
    tree_thetas = vision.get_thetas(tree_centroids)

    # Gets the latitude, longitudes of the trees detected
    tree_lat_lons = vision.scan_image_sat(satellite_image)

    # Initialize the xys of the trees in the satellite image
    tree_xys = []

    # Go through all the coordinates in the tree_lat_lons and convert them to x y
    for coordinate in tree_lat_lons:
        tree_xys.append(converter.latlon_to_xy(coordinate[0], coordinate[1]))
    
    # Initialize the tree theta primes
    tree_theta_primes = []

    # Go through all the thetas in the tree_thetas
    for theta in tree_thetas:

        # Add the theta to the current heading
        transformed_theta = heading + theta
        
        # Check if the transformed theta is greater than 360
        # If true, decrease it by 360
        # Else if less than 0, increase it by 360
        if transformed_theta > 360:
            transformed_theta -= 360.0
        
        elif transformed_theta < 0:
            transformed_theta += 360.0
        
        # Appends the adjusted theta to the tree theta primes
        tree_theta_primes.append(transformed_theta)
    
    # Initialize the slopes
    slopes = []

    # Loops through all of the tree theta primes to get their slopes
    for theta_prime in tree_theta_primes:

        # Calculates slope m from the theta
        m = math.sin(math.radians(theta_prime)) / math.cos(math.radians(theta_prime))

        # Appends the slope to slopes
        slopes.append(m)

    # Initializes the y-intercepts
    bs = []

    # Loops through all the tree xys
    for i in range(len(tree_xys)):

        # Calculate the b value from the x, y, and slope
        b = tree_xys[i][1] - slopes[i] * tree_xys[i][0]

        # Append the b to the bs
        bs.append(b)

    # Loop through all of the tree xys twice
    for i in range(len(tree_xys)):
        for j in range(len(tree_xys)):

            # Check if the i and j are not the same
            if i != j:

                # Get the slope and y_intercept of tree 1
                tree1_m = slopes[i]
                tree1_b = bs[i]

                # Get the slope and y intercept of tree 2
                tree2_m = slopes[j]
                tree2_b = bs[j]

                # Get their x and y intercepts
                x_int = (tree2_b - tree1_b) / (tree1_m - tree2_m)
                y_int = x_int * tree1_m + tree1_b

                # Convert the x and y to lat lon
                lat, lon = converter.xy_to_latlon(x_int, y_int)

                # Get the correct lat and lon
                correct_lat, correct_lon = converter.latlon_to_xy(correct_point[0], correct_point[1])

                print("GPS Coord: ", converter.latlon_to_xy(gps_coord[0], gps_coord[1]))
                print("Tree thetas: ", tree_thetas)
                print("Tree theta primes: ", tree_theta_primes)
                print("Tree 1 xy: ", str(tree_xys[0][0]) + ', ' + str(tree_xys[0][1]))
                print("Tree 2 xy: ", str(tree_xys[1][0]) + ', ' + str(tree_xys[1][1]))
                print("Intersection point: ", str(x_int) + ', ' + str(y_int))
                print("Correct lat lon: ", str(correct_lat) + ', ' + str(correct_lon))
                print("Predicted lat lon: ", str(lat) + ", " + str(lon))
                print("Error: " + str(converter.haversine(lat, lon, correct_point[0], correct_point[1])))

if __name__ == '__main__':

    vision = Vision()

    converter = Converter(35.7713528, -78.673756)

    correct_point = [35.77095751, -78.6751402]

    main()
