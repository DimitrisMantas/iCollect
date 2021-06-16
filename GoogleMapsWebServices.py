"""iCollect - Intelligent Waste Collection Vehicle Routing"""


######################################################################
#                           LINTER SETTINGS                          #
######################################################################
# pylint: disable = consider-using-enumerate                         #
# pylint: disable = invalid-name                                     #
# pylint: disable = line-too-long                                    #
# pylint: disable = wrong-import-position                            #
# pylint: disable = wrong-import-order                               #
######################################################################


from __future__ import division
from __future__ import print_function

import codecs
import csv
import json
import pickle

import html2text
import polyline
import gmplot
import numpy as np

import time
import urllib.request


def build_data_frame(BinCapacity, FillLevelThreshold):
    """Build the required data frame."""

    def load_credentials():
        """Load the Google Web Services API key from file."""
        with open(".KEEP_API_KEY.txt", "r") as API_KEY:
            API_KEY = API_KEY.read()
        return API_KEY

    def reduce_location_list():
        """ """

        def build_location_list():
            """ Get the API key for the Google Distance Matrix API from its corresponding file."""
            location_list = []
            with open("BinInformation.csv", "rb") as BinInformation:
                reader = csv.reader(codecs.iterdecode(BinInformation, "utf-8"), delimiter=",")
                # Skip the first line of the file.
                next(reader)
                for row in reader:
                    location = float(row[6]), float(row[7])
                    BinLoad = int(row[3])
                    if BinLoad == 0 or BinLoad >= BinCapacity * FillLevelThreshold:
                        location_list.append(location)
            return location_list

        # Build the location list as a NumPy structure.
        LocationList = np.array(build_location_list())

        # The new matrix is sorted in alphabetical/numerical order.
        # To preserve the original location order, we have to get the smallest index of each unique
        # location and then sort according to them.
        _, index = np.unique(LocationList, axis=0, return_index=True)
        UniqueLocations = LocationList[np.sort(index)]

        reduced_location_list = list(
            map(tuple, UniqueLocations)
        )  # Convert the NumPy structure to a list of tuples as in the original structure.

        return reduced_location_list

    data_frame = {}

    data_frame["API_KEY"] = load_credentials()
    data_frame["location_list"] = reduce_location_list()

    return data_frame


def make_handshake(origin_location, destination_location, waypoints, DepartureTime, api, API_KEY):
    """Build and send an HTTPS request to the Google Distance Matrix API to provide the required
    information and receive its response in the form of a .JSON file."""

    def build_request(locations, location_mode):
        """Convert the list of locations into a pipe separated list."""

        # The `origin` and `destination` parameters do not need a `via:` option.
        # The same is true for the `waypoint` parameter if `departure _time` is not defined.
        if location_mode == "route_leg" or (location_mode == "route_influencer" and DepartureTime == 0):
            location_string = ""
            for i in range(len(locations) - 1):
                location_string += str(locations[i][0]) + "," + str(locations[i][1]) + "|"

            location_string += str(locations[-1][0]) + "," + str(locations[-1][1])
        elif location_mode == "route_influencer" and DepartureTime > 0:
            location_string = "via:"
            for i in range(len(locations) - 1):
                location_string += str(locations[i][0]) + "," + str(locations[i][1]) + "|via:"

            location_string += str(locations[-1][0]) + "," + str(locations[-1][1])

        return location_string

    if DepartureTime == 0:
        # When the travel time slack is equal to zero, then there is no need to specify a departure
        # time for the Google Distance Maatrix API.
        #
        # This also means that the choice of route and the corresponding travel time are based on
        # road network and average time-independent traffic conditions, i.e. no live traffic model
        # is called.
        # https://developers.google.com/maps/documentation/distance-matrix/overview#optional-parameters

        if api == "GoogleDistanceMatrixAPI":
            request = "https://maps.googleapis.com/maps/api/distancematrix/json?"
        elif api == "GoogleDirectionsAPI":
            request = "https://maps.googleapis.com/maps/api/directions/json?"
    elif DepartureTime == 1:
        if api == "GoogleDistanceMatrixAPI":
            request = "https://maps.googleapis.com/maps/api/distancematrix/json?departure_time=now&"
        elif api == "GoogleDirectionsAPI":
            request = "https://maps.googleapis.com/maps/api/directions/json?departure_time=now&"
    else:
        # The time before the start of the trip in minutes.
        DepartureTime = int(round(time.time()) + (DepartureTime * 60))

        if api == "GoogleDistanceMatrixAPI":
            request = (
                "https://maps.googleapis.com/maps/api/distancematrix/json?departure_time=" + str(DepartureTime) + "&"
            )
        elif api == "GoogleDirectionsAPI":
            request = "https://maps.googleapis.com/maps/api/directions/json?departure_time=" + str(DepartureTime) + "&"

    if api == "GoogleDistanceMatrixAPI":
        origin_location_string = build_request(origin_location, location_mode="route_leg")
        destination_location_string = build_request(destination_location, location_mode="route_leg")
    elif api == "GoogleDirectionsAPI":
        origin_location_string = build_request(origin_location, location_mode="route_leg")
        destination_location_string = build_request(destination_location, location_mode="route_leg")
        waypoints_string = build_request(waypoints, location_mode="route_influencer")

    if api == "GoogleDistanceMatrixAPI":
        request += (
            "origins=" + origin_location_string + "&destinations=" + destination_location_string + "&key=" + API_KEY
        )
    elif api == "GoogleDirectionsAPI":
        request += (
            "origin="
            + origin_location_string
            + "&destination="
            + destination_location_string
            + "&waypoints="
            + waypoints_string
            + "&key="
            + API_KEY
        )

    response = urllib.request.urlopen(request).read()
    response = json.loads(response)
    return response


def build_distance_matrix(response):
    """Build the travel distance matrix for the list of locations."""
    distance_matrix = []

    for row in response["rows"]:
        row_list = [row["elements"][i]["distance"]["value"] for i in range(len(row["elements"]))]

        distance_matrix.append(row_list)

    return distance_matrix


def build_time_matrix(response, DepartureTime):
    """Build the travel time matrix for the list of locations."""
    time_matrix = []

    for row in response["rows"]:
        if DepartureTime == 0:
            row_list = [row["elements"][i]["duration"]["value"] for i in range(len(row["elements"]))]
        else:
            row_list = [row["elements"][i]["duration_in_traffic"]["value"] for i in range(len(row["elements"]))]

        time_matrix.append(row_list)
    return time_matrix


def build_directions(response, num_locations, DepartureTime):
    """ """
    directions = ""

    if DepartureTime == 0:
        for i in range(num_locations):
            for ii in range(len(response["routes"][0]["legs"][i]["steps"])):
                # TODO - HTML2TEXT should be called once.
                directions += html2text.html2text(response["routes"][0]["legs"][i]["steps"][ii]["html_instructions"])
            directions += html2text.html2text("\n")

    else:
        for i in range(len(response["routes"][0]["legs"][0]["steps"])):
            directions += html2text.html2text(response["routes"][0]["legs"][0]["steps"][i]["html_instructions"])

    return directions


def save_distance_matrix(distance_matrix):
    """Serialize the travel distance and time matrices for the list of locations to disk for later use."""
    with open(".KEEP_DISTANCE_MATRIX.txt", "wb") as DISTANCE_MATRIX:
        pickle.dump(distance_matrix, DISTANCE_MATRIX)


def save_time_matrix(time_matrix):
    with open(".KEEP_TIME_MATRIX.txt", "wb") as TIME_MATRIX:
        pickle.dump(time_matrix, TIME_MATRIX)


def save_directions(directions, route_index):
    """Serialize the travel distance and time matrices for the list of locations to disk for later use."""
    with open(".OUT_ROUTE_" + str(route_index) + ".txt", "a", encoding="utf-8") as OUTPUT_FILE:
        OUTPUT_FILE.write(directions)


def build_pline(response):
    # Get the overview polyline.
    pline = response["routes"][0]["overview_polyline"]["points"]
    # Decode the overview polyline into a list of points.
    pline = polyline.decode(pline)
    return pline


def draw_map(num_locations, response, waypoints, API_KEY, ROUTE_LEG_INDEX, block_list, route_index, pline):
    """ """

    # If you need iCollect to build the overview polyline for you, you have to specify "build_pline".
    # If you alreasy have the polyline as a list of tuples, just provide it
    if pline == "build_pline":
        # Build the overview polyline.
        pline = build_pline(response)

    # Zip the pline.
    pline = zip(*pline)

    # THIS LINE IS LOCATION SPECIFIC.
    gmap = gmplot.GoogleMapPlotter(38.246640, 21.734574, 13, apikey=API_KEY)

    # Plot the overview pline.
    # The pline decoding algorithm works with 5 significant digits, so the extra precision is wasted.
    gmap.plot(*pline, edge_width=3, color="#1E90FF", precision=5)  # #1E90FF -> Dodger Blue

    # Plot the markers and the circles. The vehicle depot needs to snap to the polyline.
    for i in range(num_locations):
        if i == 0:
            marker_latitude = 38.24557
            marker_longitude = 21.78249
            color = "#90EE90"  # #90EE90 -> Light Green
        else:
            marker_latitude = waypoints[i][0]
            marker_longitude = waypoints[i][1]
            color = "#DC143C"  # #DC143C -> Crimson

        marker_label = block_list[ROUTE_LEG_INDEX[i]]

        gmap.marker(marker_latitude, marker_longitude, color=color, label=marker_label, precision=5)
        if i != 0:
            gmap.circle(
                marker_latitude, marker_longitude, 1.11
            )  # This is the average precision of 5-digit coordinates at the equator.

    # Save the map to an .HTML file.
    gmap.draw(".OUT_ROUTE_" + str(route_index) + ".html")


def GoogleDistanceMatrixAPI(BinCapacity, DepartureTime, FillLevelThreshold):
    """Build the entry point of the program."""
    # Build the data frame.
    data_frame = build_data_frame(BinCapacity, FillLevelThreshold)

    API_KEY = data_frame["API_KEY"]

    location_list = data_frame["location_list"]
    num_locations = len(location_list)
    # This is a guide to the most relevant publicly available Google Distance Matrix API billing and
    # usage information.
    # https://developers.google.com/maps/documentation/distance-matrix/
    # https://developers.google.com/maps/documentation/distance-matrix/web-service-best-practices
    # https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing
    #
    # There is also 90-day trial period, which includes $300 in free Cloud Billing credits for all
    # new Google Cloud and Google Maps Platform users.
    # https://cloud.google.com/free/docs/gcp-free-tier#free-trial
    #
    # The total number of origin and destination locations is hard-limited to 25.
    max_num_locations = 10

    # In case the total number of locations is less than the hard limit, we can go ahead and send the request as it is.
    if num_locations <= max_num_locations:

        distance_matrix = []

        time_matrix = []

        response = make_handshake(
            location_list,
            location_list,
            waypoints="",
            DepartureTime=DepartureTime,
            api="GoogleDistanceMatrixAPI",
            API_KEY=API_KEY,
        )

        distance_matrix += build_distance_matrix(response)
        time_matrix += build_time_matrix(response, DepartureTime)

        save_distance_matrix(distance_matrix)
        save_time_matrix(time_matrix)

    else:
        # Split the NxN travel distance and time matrices into q 10x10 chunks, as well as one r x (q * 10), one (q * 10) x r and one r x r chunk, which will be appropriately concatenated later.
        q, r = divmod(num_locations, max_num_locations)

        # The the travel distance and time matrices are defined as two square integer matrices of the appropriate shape and are filled with zeros.
        #
        # Each sub-matrix will be placed at the correct position to replace to zeros.

        def build_iteration_index(iterating_var, reference_var):
            min_index = iterating_var * reference_var
            max_index = (iterating_var + 1) * reference_var

            return min_index, max_index

        full_distance_matrix = np.empty([num_locations, num_locations], dtype=int)
        full_time_matrix = np.empty([num_locations, num_locations], dtype=int)

        for i in range(q):
            # Split the rows of the matrices into q chunks of 10.
            min_index_i = build_iteration_index(i, max_num_locations)[0]
            max_index_i = build_iteration_index(i, max_num_locations)[1]

            origin_locations = location_list[min_index_i:max_index_i]

            for j in range(q):
                # For each group of rows, get q chucks of 10 columns.
                min_index_j = build_iteration_index(j, max_num_locations)[0]
                max_index_j = build_iteration_index(j, max_num_locations)[1]

                destination_locations = location_list[min_index_j:max_index_j]

                response = make_handshake(
                    origin_locations,
                    destination_locations,
                    waypoints="",
                    DepartureTime=DepartureTime,
                    api="GoogleDistanceMatrixAPI",
                    API_KEY=API_KEY,
                )

                distance_matrix = build_distance_matrix(response)
                full_distance_matrix[min_index_i:max_index_i, min_index_j:max_index_j] = distance_matrix

                time_matrix = build_time_matrix(response, DepartureTime)
                full_time_matrix[min_index_i:max_index_i, min_index_j:max_index_j] = time_matrix

                time.sleep(0.1)

        if r > 0:
            # Build the r x (q * 10) submatrix.
            # Continue from where we left off and basically do the same thing one more time.

            min_index_q = q * max_num_locations

            origin_locations = location_list[min_index_q:num_locations]

            for i in range(q):
                min_index_i = build_iteration_index(i, max_num_locations)[0]
                max_index_i = build_iteration_index(i, max_num_locations)[1]

                destination_locations = location_list[min_index_i:max_index_i]

                response = make_handshake(
                    origin_locations,
                    destination_locations,
                    waypoints="",
                    DepartureTime=DepartureTime,
                    api="GoogleDistanceMatrixAPI",
                    API_KEY=API_KEY,
                )

                distance_matrix = build_distance_matrix(response)

                full_distance_matrix[min_index_q:num_locations, min_index_i:max_index_i] = distance_matrix

                time_matrix = build_time_matrix(response, DepartureTime)

                full_time_matrix[min_index_q:num_locations, min_index_i:max_index_i] = time_matrix

                time.sleep(0.1)

            # Build the (q * 10) * r submatrix.
            for i in range(q):

                min_index_i = build_iteration_index(i, max_num_locations)[0]
                max_index_i = build_iteration_index(i, max_num_locations)[1]

                origin_locations = location_list[min_index_i:max_index_i]

                destination_locations = location_list[min_index_q:num_locations]

                response = make_handshake(
                    origin_locations,
                    destination_locations,
                    waypoints="",
                    DepartureTime=DepartureTime,
                    api="GoogleDistanceMatrixAPI",
                    API_KEY=API_KEY,
                )

                distance_matrix = build_distance_matrix(response)

                full_distance_matrix[min_index_i:max_index_i, min_index_q:num_locations] = distance_matrix

                time_matrix = build_time_matrix(response, DepartureTime)
                full_time_matrix[min_index_i:max_index_i, min_index_q:num_locations] = time_matrix

                time.sleep(0.1)

            # Build the r x r submatrix.
            origin_locations = location_list[min_index_q:num_locations]
            destination_locations = origin_locations

            response = make_handshake(
                origin_locations,
                destination_locations,
                waypoints="",
                DepartureTime=DepartureTime,
                api="GoogleDistanceMatrixAPI",
                API_KEY=API_KEY,
            )

            distance_matrix = build_distance_matrix(response)
            full_distance_matrix[min_index_q:num_locations, min_index_q:num_locations] = distance_matrix

            time_matrix = build_time_matrix(response, DepartureTime)
            full_time_matrix[min_index_q:num_locations, min_index_q:num_locations] = time_matrix

        full_distance_matrix = full_distance_matrix.tolist()

        # When working with live traffic conditions, Google's APIs can sometimes return a time matrix with non-zero diagonal elements.
        # This is an artifact and must be fixed.
        T = np.array(full_time_matrix)
        np.fill_diagonal(T, 0)
        full_time_matrix = T.tolist()

        save_distance_matrix(full_distance_matrix)
        save_time_matrix(full_time_matrix)


# pylint: disable=invalid-name
def GoogleDirectionsAPI(BinCapacity, block_list, DepartureTime, FillLevelThreshold):
    """Build the entry point of the program."""
    # Build the data frame.
    data_frame = build_data_frame(BinCapacity, FillLevelThreshold)

    API_KEY = data_frame["API_KEY"]
    location_list = data_frame["location_list"]

    # The number of route visits (i.e. building blocks) must be no greater than 25.
    MAX_ROUTE_WAYPOINTS = 25

    with open(".PARSE_VEHICLE_INDEX.txt", "rb") as VEHICLE_INDEX:
        VEHICLE_INDEX = pickle.load(VEHICLE_INDEX)

    NUM_ROUTES = len(VEHICLE_INDEX)

    # For each route...
    for i in range(NUM_ROUTES):
        # Load the .PARASE_ROUTE_LEG_LIST.
        with open(".PARSE_ROUTE_LEG_LIST_" + str(VEHICLE_INDEX[i]) + ".txt", "rb") as ROUTE_LEG_INDEX:
            ROUTE_LEG_INDEX = pickle.load(ROUTE_LEG_INDEX)

        # Build the route leg coordinates from the corresponding indices. # THIS CONTAINS ALL THE LEGS INCLUDING THE TWO ZEROS.
        route_leg_coordinates = []
        for ii in range(len(ROUTE_LEG_INDEX)):
            route_leg_coordinates.append(location_list[ROUTE_LEG_INDEX[ii]])

        # TODO - WHATEVER IS NAMED "LEG(S).." REFERS TO WHOLE ROUTE (LANDFILL + BLOCKS + LANDFILL).
        # TODO - WHATEVER IS NAMED "LOCATIONS..." REFERS TO THE BUILDING BLOCKS + THE LANDFILL.
        # TODO - WHATEVER IS NAMED "WAYPOINTS..." REFERS TO THE BUILDING BLOCKS
        # TODO - THE "NUMBER OF LOCATIONS PASSED" TO THE APIS IS THE NUMBER OF THE BUILDING BLOCKS + LANDFILL

        # Check if the length of the route is within spec.
        # Building blocks + landfill. We can do only the waypoints,but this particular len is useful for later and doesn't really make anything harder than it needs to be.
        # Just remember to add  a +1 to max_locations to account for the landfill.
        num_locations = len(route_leg_coordinates) - 1

        if num_locations <= (MAX_ROUTE_WAYPOINTS + 1):
            # Define the origin and the destination locations and remove them from "route_leg_coordinates".

            # We are not popping the origin because we need it for drawing.
            origin = [route_leg_coordinates[0]]
            destination = [route_leg_coordinates.pop(-1)]

            # Whatever is still left represents the route waypoints.
            waypoints = route_leg_coordinates

            response = make_handshake(
                origin,
                destination,
                waypoints=waypoints,
                DepartureTime=DepartureTime,
                api="GoogleDirectionsAPI",
                API_KEY=API_KEY,
            )
            directions = build_directions(response, num_locations + 1, DepartureTime)
            save_directions(directions, VEHICLE_INDEX[i])

            draw_map(
                num_locations,
                response,
                waypoints,
                API_KEY,
                ROUTE_LEG_INDEX,
                block_list,
                VEHICLE_INDEX[i],
                pline="build_pline",
            )

        else:
            # Compute the number of splits.
            num_cuts = np.ceil(
                num_locations / MAX_ROUTE_WAYPOINTS
            )  # This splits with respect to Google's obscure limitations, but also leaves room to add the origin and location.

            # Split using NumPY. Insert the documentation about the splitting procedure and all that stuff.
            # This gets the list of tuples we give it and produces a structures similar to a list of lists of lists, so this monstrosity will bring the correct structure back.
            # Basically we give Numpy something like [(...,...),(...,...),...] and it returns something like
            # [array([...,...],[...,...],...]), array([...,...],[...,...],...]),...]. The only way this is coming back to the original structure is like so
            # [array([...,...],[...,...],...]), array([...,...],[...,...],...]),...] -> [[[...,...],[...,...],...],[[...,...],[...,...],...]] -> [[(...,...),(...,...),...],[(...,...),(...,...),...]]
            # Technically, lists and tuples have the same indexing notation, so this might be unessecary but whatever.
            stupid_numpy_structure = np.array_split(route_leg_coordinates, num_cuts)

            sub_route_leg_coordinates = []

            # Register the route pline.
            nested_route_pline = []

            # For each NumPy array in the resulting list...
            for iii in range(len(stupid_numpy_structure)):

                correct_structure = stupid_numpy_structure[iii].tolist()  # Convert said array to a Python list.
                correct_structure = list(
                    map(tuple, correct_structure)
                )  # Map the resulting list to a tuple. This produces a list of tuples.

                # Add the correct thing back where it belongs.
                sub_route_leg_coordinates.append(correct_structure)

                # Enough bulshit. Let's do this..
                # The very first sub-route contains both its origin and destination.
                # The following sub-routes contain only their destination and their origin is the previous sub-route's destination.
                if iii == 0:

                    origin = [sub_route_leg_coordinates[iii][0]]
                    destination = [sub_route_leg_coordinates[iii][-1]]

                    # This gets everything from the second element (inclusive) up to the last element (non-inclusive).
                    waypoints = sub_route_leg_coordinates[iii][1:-1]

                else:

                    origin = [sub_route_leg_coordinates[iii - 1][-1]]
                    destination = [sub_route_leg_coordinates[iii][-1]]

                    waypoints = sub_route_leg_coordinates[iii][0:-1]

                response = make_handshake(
                    origin,
                    destination,
                    waypoints=waypoints,
                    DepartureTime=DepartureTime,
                    api="GoogleDirectionsAPI",
                    API_KEY=API_KEY,
                )
                directions = build_directions(response, len(waypoints) + 1, DepartureTime)
                save_directions(directions, VEHICLE_INDEX[i])

                # Decode sub-route polyline and append it to the route polyline.
                # Due to the nature of Google's Polyline Algo, we can just append them all together and then decode the final product. Decoding need to take place one by one.
                sub_route_pline = build_pline(response)
                nested_route_pline.append(sub_route_pline)

            # Appending creates a list of lists where each nested list is the sub-route pline.
            # Flatten.
            route_pline = []
            for sub_route_pline in nested_route_pline:
                for coordinates in sub_route_pline:
                    route_pline.append(coordinates)

            # The problem is with getting directions. There is no known hard limit on the amount of markers.
            route_waypoints = route_leg_coordinates[0:-1]

            # Draw the map here.
            draw_map(
                num_locations,
                response,
                route_waypoints,
                API_KEY,
                ROUTE_LEG_INDEX,
                block_list,
                VEHICLE_INDEX[i],
                pline=route_pline,
            )
