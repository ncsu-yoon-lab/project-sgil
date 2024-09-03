from vision import Vision
from converter import Converter
import math

def main():
    vehicle_image = vision.get_image_vehicle()

    # Only for testing purposes
    gps_coord = [35.77093917, -78.6750895, 191.4499969]

    heading = gps_coord[2]

    satellite_image = vision.get_image_sat(gps_coord)

    tree_centroids = vision.scan_image_vehicle(vehicle_image)

    tree_thetas = vision.get_thetas(tree_centroids)

    tree_lat_lons = vision.scan_image_sat(satellite_image)

    tree_xys = []

    for coordinate in tree_lat_lons:
        tree_xys.append(converter.latlon_to_xy(coordinate[0], coordinate[1]))
    
    tree_theta_primes = []

    for theta in tree_thetas:

        transformed_theta = heading + theta
        
        if transformed_theta > 180:
            theta_prime = transformed_theta - 180.0
        else:
            theta_prime = transformed_theta + 180.0
        
        tree_theta_primes.append(theta_prime)
    
    slopes = []

    for theta_prime in tree_theta_primes:
        m = math.sin(math.radians(theta_prime)) / math.cos(math.radians(theta_prime))

        slopes.append(m)

    bs = []

    for i in range(len(tree_xys)):
        b = tree_xys[i][1] - slopes[i] * tree_xys[i][0]

        bs.append(b)

    for i in range(len(tree_xys)):
        for j in range(len(tree_xys)):
            if i != j:
                tree1_m = slopes[i]
                tree1_b = bs[i]

                tree2_m = slopes[j]
                tree2_b = bs[j]

                x_int = (tree2_b - tree1_b) / (tree1_m - tree2_m)
                y_int = x_int * tree1_m + tree1_b

                lat, lon = converter.xy_to_latlon(x_int, y_int)

                correct_lat, correct_lon = converter.latlon_to_xy(correct_point[0], correct_point[1])
                print(tree_thetas)
                print(tree_theta_primes)
                print(str(tree_xys[0][0]) + ', ' + str(tree_xys[0][1]))
                print(str(tree_xys[1][0]) + ', ' + str(tree_xys[1][1]))
                print(str(x_int) + ', ' + str(y_int))
                print(str(correct_lat) + ', ' + str(correct_lon))
                print(str(lat) + ", " + str(lon))
                print("Error: " + str(converter.haversine(lat, lon, correct_point[0], correct_point[1])))

if __name__ == '__main__':

    vision = Vision()

    converter = Converter(35.7713528, -78.673756)

    correct_point = [35.77095751, -78.6751402]

    main()
