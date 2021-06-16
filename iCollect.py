"""iCollect - Intelligent Waste Collection Vehicle Routing"""


######################################################################
#                          LINTER SETTINGS                           #
######################################################################
# pylint: disable=consider-using-enumerate                           #
# pylint: disable=invalid-name                                       #
# pylint: disable=line-too-long                                      #
# pylint: disable=no-member                                          #
######################################################################


from __future__ import division
from __future__ import print_function

import codecs
import csv
import pickle
import time

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# from GoogleMapsWebServices import GoogleDistanceMatrixAPI
from GoogleMapsWebServices import GoogleDirectionsAPI


def build_data_frame(BinCapacity, FillLevelThreshold, DepartureTimeFromMinutes, ArrivalTimeFromMinutes):
    """Build the required data frame."""

    # Turn this on to rebuild the travel distance and time matrices.
    # GoogleDistanceMatrixAPI(BinCapacity, DepartureTimeFromMinutes, FillLevelThreshold)

    def load_distance_matrix():
        """Load the travel distance matrix from file."""
        with open(".KEEP_DISTANCE_MATRIX.txt", "rb") as DISTANCE_MATRIX:
            distance_matrix = pickle.load(DISTANCE_MATRIX)
        return distance_matrix

    def load_time_matrix():
        """Load the travel time matrix from file."""
        with open(".KEEP_TIME_MATRIX.txt", "rb") as TIME_MATRIX:
            travel_time_matrix = pickle.load(TIME_MATRIX)
        return travel_time_matrix

    # It is inefficient to read the same file multiple times.
    def load_bin_list():

        """Load the bin list from file with respect to the corresponding building blocks."""
        bin_list = []

        with open("BinInformation.csv", "rb") as BinInformation:
            # These are the appropriate delimiter and encoding settings when exporting .CSV files from Microsoft Excel.
            reader = csv.reader(codecs.iterdecode(BinInformation, "utf-8"), delimiter=",")

            # Skip the first line of the file.
            next(reader)

            for row in reader:
                BinLoad = int(row[3])  # The variable format inside a .CSV file is undefined.
                BinList = int(row[5])

                # Include the landfill and all bins whose fill level is above the predefined threshold.
                if BinLoad == 0 or BinLoad >= BinCapacity * FillLevelThreshold:
                    bin_list.append(BinList)
        return bin_list

    def build_load_list():
        """Load and group the bin load list from file with respect to the corresponding building blocks."""

        def load_load_list():
            """Load the bin load list from file with respect to the corresponding building blocks."""
            load_list = []

            with open("BinInformation.csv", "rb") as BinInformation:
                reader = csv.reader(codecs.iterdecode(BinInformation, "utf-8"), delimiter=",")

                next(reader)

                for row in reader:
                    BinLoad = int(row[3])

                    if BinLoad == 0 or BinLoad >= BinCapacity * FillLevelThreshold:
                        load_list.append(BinLoad)
            return load_list

        # Register the bin load list as a NUMPY data structure.
        LoadList = np.array(load_load_list())

        def load_block_list():
            """Group the bin load list from file with respect to the corresponding building blocks."""
            # This is a list of the building blocks.
            block_list = []

            with open("BinInformation.csv", "rb") as BinInformation:
                reader = csv.reader(codecs.iterdecode(BinInformation, "utf-8"), delimiter=",")

                next(reader)

                for row in reader:
                    BinLoad = int(row[3])
                    BlockID = int(row[4])

                    if BinLoad == 0 or BinLoad >= BinCapacity * FillLevelThreshold:
                        block_list.append(BlockID)
            return block_list

        BlockList = np.array(load_block_list())

        # The new matrix is sorted in alphabetical/numerical order. In order to preserve the original file order, we have to get the first index of each building block and then sort according to it.
        _, index = np.unique(BlockList, axis=0, return_index=True)
        UniqueLocations = BlockList[np.sort(index)]

        # Combine the bin load list with the building block list into a 2xN matrix.
        TempMatrix = np.array([LoadList, BlockList])

        # For each building block, get its column indices in TempMatrix and sum the corresponding loads.
        load_list = []

        for i in UniqueLocations:
            index = np.where(TempMatrix[1, :] == i)[0]

            load = 0

            for ii in range(len(index)):
                load += LoadList[index[ii]]

            load_list.append(load)

        # Register the bin load list as a PYTHON data structure.
        block_list = UniqueLocations.tolist()
        return block_list, load_list, TempMatrix  # TempMatrix is quite useful for some later operations.

    def load_vehicle_cap_list():
        """Load the maximum vehicle load list from file with respect to the corresponding building blocks."""
        vehicle_cap_list = []

        with open("VehicleInformation.csv", "rb") as VehicleInformation:
            reader = csv.reader(codecs.iterdecode(VehicleInformation, "utf-8"), delimiter=",")

            next(reader)

            for row in reader:
                VehicleCap = int(row[1])
                vehicle_cap_list.append(VehicleCap)
        return vehicle_cap_list

    # Register the required data frame.
    data_frame = {}

    data_frame["distance_matrix"] = load_distance_matrix()
    data_frame["travel_time_matrix"] = load_time_matrix()

    data_frame["bin_list"] = load_bin_list()  # This all the bins before they are grouped.
    # This is how to obtain a specific output from a function with multiple outputs.
    data_frame["block_list"] = build_load_list()[0]
    # This is required information by Google OR-Tools.
    data_frame["depot"] = data_frame["block_list"][0]

    data_frame["load_list"] = build_load_list()[1]

    data_frame["vehicle_capacities"] = load_vehicle_cap_list()
    # This is required information by Google OR-Tools.
    data_frame["num_vehicles"] = len(data_frame["vehicle_capacities"])

    data_frame["TempMatrix"] = build_load_list()[2]

    def build_cost_matrix():
        """Build the the travel cost matrix according to the predefined objective function."""
        T = np.array(data_frame["travel_time_matrix"])
        V = np.array(data_frame["load_list"])

        # The travel cost between building block is equal to the corresponding travel time plus the service time of the origin block multiplied be the corresponding bin load. This means that it is bad to be moving a "big" load for a "long" period of time.

        # STEP 1 - Compute the number of bins in each building block and their effective fill levels. This means that, no matter its actual value, a bin group load can be "split" into Q "full" bin loads and one HF "half full" bin load.
        F, HF = divmod(V, BinCapacity)
        HF = HF / BinCapacity

        # STEP 2 - Compute the service time of each building block.
        # The service time of each bin with respect to its fill level, F, is equal to S(F) = 105.8823529 * F + 14.11764706, where F ∈ [0.15, 1].
        def service_time(FillLevel):
            """Compute the service time of one bin with respsect to its fill level."""
            # This implemention of the bin service time function returns wrong results for the landfill, because S(0) = 14.11764706. Also, there is no need to check its upper bound, because bin loads which are greater than BinCapacity are corrected to BinCapacity.
            service_time = 105.8823529 * FillLevel + 14.11764706
            return service_time

        # The service time of each building block is equal to Q * S(1) + S(R). All values of S are rounded to the nearest integer.
        S = np.around(F * service_time(1) + service_time(HF), 0)
        # The original variable format in S is preserved. Convert all of its contents to integers.
        S = S.astype(int)

        # STEP 3 - Compute the travel and service time between two building blocks.
        # The service time of the landfill should be equal to zero. Focus the application of the following operation to the building blocks.
        T[1:, :] = (T[1:, :].T + S[1:]).T
        # The distance between two identical building blocks should be equal to zero. Fill the diagonal of T with zeros.
        np.fill_diagonal(T, 0)

        # STEP 4 - Compute the travel cost time between two building blocks.
        # Compute the Hadamard Pr
        # oduct of T with V. Their product is not defined so the similar one-liner for T and S doesn't work, because it is not subscriptable.
        C = (T.T * V).T
        # The travel cost between the landfill and some building b
        # lock should be equal to the corresponding travel and service time, because V(0) = 0. Replace the first row of C with the first row of T.
        C[0, :] = T[0, :]
        cost_matrix = C.tolist()

        # STEP 5 - Replace the travel time matrix with the new one in order to obtain correct routing_model_solution statistics.
        travel_time_matrix = T.tolist()

        # Correct the value of the first element of S to zero and add another zero in the end to indicate the return to the landfill.
        S[0] = 0
        S = np.append(S, 0)
        service_time_matrix = S.tolist()
        return cost_matrix, travel_time_matrix, service_time_matrix

    data_frame["cost_matrix"] = build_cost_matrix()[0]
    data_frame["travel_time_matrix"] = build_cost_matrix()[1]
    data_frame["service_time_matrix"] = build_cost_matrix()[2]

    # Each location must be serviced within a specified time window, so that that the duration of each route is no longer than 8 hours or 288800 seconds.

    # Build and register said time windows.
    data_frame["num_locations"] = len(data_frame["block_list"])

    working_hours = (0, ArrivalTimeFromMinutes * 60)

    data_frame["time_windows"] = []
    for _ in range(data_frame["num_locations"]):
        data_frame["time_windows"].append(working_hours)

    return data_frame


####################################################################################################
def print_solution(data_frame, routing_model_index_manager, routing_model, routing_model_solution):
    """Print each route of the routing_model_solution of the routing model to both the terminal window and some corresponding .TXT files. Include some interesting routing_model_solution statistics."""
    # Initiate variables for some interesting routing model solution statistics.
    min_route_length = 0
    min_route_duration = 0
    min_route_service_time = 0
    min_route_load = 0

    max_route_length = 0
    max_route_duration = 0
    max_route_service_time = 0
    max_route_load = 0

    sum_route_length = 0
    sum_route_duration = 0
    sum_route_service_time = 0
    sum_route_load = 0

    # Initialize .PARSE_VEHICLE_INDEX.
    VEHICLE_INDEX = []

    # Begin to route each vehicle.
    for vehicle_index in range(data_frame["num_vehicles"]):
        # Register the index of each route leg.
        route_leg_index = routing_model.Start(vehicle_index)

        # This is what gets printed to the terminal window and the corresponding .TXT files.
        route = "-------------------\n" + "ROUTE FOR VEHICLE {}\n".format(vehicle_index + 1) + "-------------------\n"
        # This is what gets passed to the Directions and Maps JavaScript APIs.
        ROUTE_LEG_LIST = []  # A list helps with differentiating between large building block numbers.

        # Register the required routing_model_solution statistics, but for this particular route.
        route_length = 0
        route_duration = 0
        route_service_time = 0
        route_load = 0

        # Begin to print the routing_model_solution for as long as the index of each route leg doesn't correspond the landfill.
        while not routing_model.IsEnd(route_leg_index):
            # Convert the index of each route leg to the corresponding OR-Tools index.
            node_index = routing_model_index_manager.IndexToNode(route_leg_index)

            # Increment route load by the appropriate bin load.
            route_load += data_frame["load_list"][node_index]

            if node_index == 0:
                route += "{0} [V = {1}] -> ".format(node_index, route_load)  # OK
            else:
                # Load TempMatrix and UniqueLocations from before.
                TempMatrix = data_frame["TempMatrix"]

                # Replace the first row of TempMatrix with the bin ID numbers.
                BinOT_ID = np.array(data_frame["bin_list"])
                TempMatrix[0, :] = BinOT_ID

                # For each unique building block, find all the bins it contains.
                bin_string = ""

                block_index = data_frame["block_list"][node_index]

                index = np.where(TempMatrix[1, :] == block_index)[0]

                bin_list = []
                for i in range(len(index)):
                    bin_list.append(BinOT_ID[index[i]])  # Bin IDs are not sorted.
                bin_list.sort()
                bin_string = str(bin_list)

                route += "{0} {1} - [V = {2} [+{3}]] -> ".format(
                    block_index, bin_string, route_load, data_frame["load_list"][node_index]
                )

            ROUTE_LEG_LIST.append(node_index)  # We are passing the nodes of the routing model to the API.

            # This is the index of the next building block in the route.
            route_leg_index = routing_model_solution.Value(routing_model.NextVar(route_leg_index))

        # While it is true that node_index = routing_model_index_manager.IndexToNode(route_leg_index), "node_index" doesn't work for some reason and even Google implements this like that.
        route += " {0} [V = {1}]\n".format(routing_model_index_manager.IndexToNode(route_leg_index), route_load) + "\n"
        ROUTE_LEG_LIST.append(routing_model_index_manager.IndexToNode(route_leg_index))

        # Compute the route statistics.
        for i in range(len(ROUTE_LEG_LIST) - 1):  # This is done, so that "i + 1" won't be out of range.
            # Find the appropriate row of the travel distance and time matrices.
            row = ROUTE_LEG_LIST[i]
            # Do the same for the column.
            column = ROUTE_LEG_LIST[i + 1]

            route_length += data_frame["distance_matrix"][row][column]
            route_duration += data_frame["travel_time_matrix"][row][column]
            route_service_time += data_frame["service_time_matrix"][column]

        # Append the route statistics to the route plan.
        route += "L = {} m\n".format(route_length)
        route += "T = {} s\n".format(route_duration)
        route += "S = {} s\n".format(route_service_time)
        route += "V = {} L\n".format(route_load)

        print(route)

        # It is possible that not all vehicles are assigned a route so iCollect outputs the ones, whose length is non-zero, i.e. the ones, which dont look like this: [0 -> 0].
        # The route numbering corresponds to the vehicle numbering, i.e. VEHICLE 5 is assigned to ROUTE 5 and so on.
        if ROUTE_LEG_LIST != [0, 0]:
            with open(".OUT_ROUTE_" + str(vehicle_index + 1) + ".txt", "w") as OUTPUT_FILE:
                print(route, file=OUTPUT_FILE)

            # The distribution of routes can be non-uniform, so we need to know exactly which route is assigned to which vehicle.
            VEHICLE_INDEX.append(vehicle_index + 1)

            with open(".PARSE_VEHICLE_INDEX.txt", "wb") as PARSE_VEHICLE_INDEX:
                pickle.dump(VEHICLE_INDEX, PARSE_VEHICLE_INDEX)

            with open(".PARSE_ROUTE_LEG_LIST_" + str(vehicle_index + 1) + ".txt", "wb") as PARSE_ROUTE_LEG_LIST:
                pickle.dump(ROUTE_LEG_LIST, PARSE_ROUTE_LEG_LIST)

        # Compute the routing_model_solution statistics.

        # It is possible that the actual minimum values of the solution statistics will be equal to zero. This information needs to be filtered out.
        min_route_length = np.array([route_length, min_route_length])
        min_route_duration = np.array([route_duration, min_route_duration])
        min_route_service_time = np.array([route_service_time, min_route_service_time])
        min_route_load = np.array([route_load, min_route_load])

        # Because all variable are under the same block, if one of the following calculations raises a ValueError, then all values are set to zero.
        try:
            min_route_length = np.min(min_route_length[np.nonzero(min_route_length)])
            min_route_duration = np.min(min_route_duration[np.nonzero(min_route_duration)])
            min_route_service_time = np.min(min_route_service_time[np.nonzero(min_route_service_time)])
            min_route_load = np.min(min_route_load[np.nonzero(min_route_load)])
        # It is posible that the first vehicle will not be assigned a route. In this case, the route minima will be undefined, and cannot be minimized further.
        # Corect the values of these statistics to zero.
        except ValueError:
            min_route_length = 0
            min_route_duration = 0
            min_route_service_time = 0
            min_route_load = 0

        # This can never be undefined.
        max_route_length = max(route_length, max_route_length)
        max_route_duration = max(route_duration, max_route_duration)
        max_route_service_time = max(route_service_time, max_route_service_time)
        max_route_load = max(route_load, max_route_load)

        # This can never be undefined.
        sum_route_length += route_length
        sum_route_duration += route_duration
        sum_route_service_time += route_service_time
        sum_route_load += route_load

    # Format the routing_model_solution statistics and print them to the terminal window.
    statistics = "-------------------\n" + "SOLUTION STATISTICS\n" + "-------------------\n"

    statistics += "min(L) = {} m\n".format(min_route_length)
    statistics += "min(T) = {} s\n".format(min_route_duration)
    statistics += "min(S) = {} s\n".format(min_route_service_time)
    statistics += "min(V) = {} L\n\n".format(min_route_load)

    statistics += "max(L) = {} m\n".format(max_route_length)
    statistics += "max(T) = {} s\n".format(max_route_duration)
    statistics += "max(S) = {} s\n".format(max_route_service_time)
    statistics += "max(V) = {} L\n\n".format(max_route_load)

    statistics += "sum(L) = {} m\n".format(sum_route_length)
    statistics += "sum(T) = {} s\n".format(sum_route_duration)
    statistics += "sum(S) = {} s\n".format(sum_route_service_time)
    statistics += "sum(V) = {} L".format(sum_route_load)

    print(statistics)


####################################################################################################
def main(BinCapacity, FillLevelThreshold, DepartureTimeFromMinutes, ArrivalTimeFromMinutes):
    """THIS IS THE ENTRY POINT OF THE PROGRAM."""
    # Build the required data frame.
    data_frame = build_data_frame(BinCapacity, FillLevelThreshold, DepartureTimeFromMinutes, ArrivalTimeFromMinutes)

    # INFORMATION - Google OR-Tools uses its own index manager.
    # While PYTHON uses indices to tranverse a data structure, GOOGLE OR-TOOLS uses nodes to tranverse a routing model.
    routing_model_index_manager = pywrapcp.RoutingIndexManager(
        data_frame["num_locations"], data_frame["num_vehicles"], data_frame["depot"]
    )

    # Register the routing model.
    routing_model = pywrapcp.RoutingModel(routing_model_index_manager)

    # Build and register the cost function callback.
    def route_cost_callback(origin_index, destination_index):
        """Return the objective cost between two nodes of the routing model."""
        # Convert PYTHON indices into GOOGLE OR-TOOLS nodes.
        origin_node = routing_model_index_manager.IndexToNode(origin_index)
        destination_node = routing_model_index_manager.IndexToNode(destination_index)
        return data_frame["cost_matrix"][origin_node][destination_node]

    route_cost_callback_index = routing_model.RegisterTransitCallback(route_cost_callback)

    # Set the objective cost of each arc of the routing model.
    routing_model.SetArcCostEvaluatorOfAllVehicles(route_cost_callback_index)

    # Build and register the bin load callback function.
    def bin_load_callback(origin_index):
        """Return the bin load to be picked-up at each node of the routing model."""
        origin_node = routing_model_index_manager.IndexToNode(origin_index)
        return data_frame["load_list"][origin_node]

    bin_load_callback_index = routing_model.RegisterUnaryTransitCallback(bin_load_callback)

    # Build and register the route duration callback function.
    def route_duration_callback(origin_index, destination_index):
        """Return the objective cost between two nodes of the routing model."""
        # Convert PYTHON indices into GOOGLE OR-TOOLS nodes.
        origin_node = routing_model_index_manager.IndexToNode(origin_index)
        destination_node = routing_model_index_manager.IndexToNode(destination_index)
        return data_frame["travel_time_matrix"][origin_node][destination_node]

    route_duration_callback_index = routing_model.RegisterTransitCallback(route_duration_callback)

    # Build and register a dimension to monitor the current vehicle fill level.
    routing_model.AddDimensionWithVehicleCapacity(
        bin_load_callback_index,
        0,  # Dimension Slack -> 0. This means that the vehicles must be emptied before leaving the landfill.
        data_frame["vehicle_capacities"],  # The maximum vehicle fill level.
        True,  # Initiate the dimension counter at zero.
        "VehicleCapacity",
    )

    # Build and register a dimension to monitor the current route duration.
    route_duration = "RouteDuration"
    routing_model.AddDimension(
        route_duration_callback_index,
        0,  # Dimension Slack -> 0. This means that the vehicles will not wait at each node.
        ArrivalTimeFromMinutes * 60,  # The maximum route duration.
        True,  # Initiate the dimension counter at zero.
        "RouteDuration",
    )

    # Add time window constraints for each node of the routing model, except the landfill.
    route_duration_dimension = routing_model.GetDimensionOrDie(route_duration)
    for location_index, time_window in enumerate(data_frame["time_windows"]):
        if location_index == 0:
            continue
        index = routing_model_index_manager.NodeToIndex(location_index)
        route_duration_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Add time window constraints for each vehicle.
    for vehicle_index in range(data_frame["num_vehicles"]):
        index = routing_model.Start(vehicle_index)
        route_duration_dimension.CumulVar(index).SetRange(
            data_frame["time_windows"][0][0], data_frame["time_windows"][0][1]
        )

    for i in range(data_frame["num_vehicles"]):
        routing_model.AddVariableMinimizedByFinalizer(route_duration_dimension.CumulVar(routing_model.Start(i)))
        routing_model.AddVariableMinimizedByFinalizer(route_duration_dimension.CumulVar(routing_model.End(i)))

    # Set the preferred strategy to find an initial routing_model_solution.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # Set the preferred local search strategy.
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # The local search strategies in Google OR-Tools have no breakpoints. Set a corresponding runtime limit.
    # search_parameters.time_limit.FromSeconds(30000)

    # Solve the routing model.
    routing_model_solution = routing_model.SolveWithParameters(search_parameters)

    # Print the routing_model_solution to a terminal window and write it to a .TXT file.
    if routing_model_solution:
        print_solution(data_frame, routing_model_index_manager, routing_model, routing_model_solution)

        block_list = data_frame["block_list"]
        GoogleDirectionsAPI(BinCapacity, block_list, DepartureTimeFromMinutes, FillLevelThreshold)

    else:
        # For more information, please refer OR-Tools search status codes.
        # This is a guide to all publicly available Google OR-Tools search status codes.
        # https://developers.google.com/optimization/routing/routing_options#search-status
        print(
            "Google OR-Tools has encountered a fatal error. The routing model could not be solved. [STATUS: {}]".format(
                routing_model.status()
            )
        )


if __name__ == "__main__":
    # This a brief guide to working with iCollect and getting the most out of its capabilities.

    # The internal units of both the program and the routing model are the following:
    # Length : [m]
    #  Time  : [s]
    # Volume : [L]

    # iCollect is generally dimensionless and the above units can be changed, but it is recommended that they remain as they are
    # You should remember that this is also the case for both inputs to the program and .OUT files, unless a unit of measurement, which is different from the ones discribed above is expicitly required.

    # Percentages are scaled within the range [0, 1], i.e. 50% -> 0.5 and so on.

    #       DepartureTimeFromMinutes -> 0        : Google OR-Tools uses cached information about the road network and average time-independent traffic conditions.
    #       DepartureTimeFromMinutes -> 1        : Google OR-Tools uses
    # DepartureTimeFromMinutes -> T, where T > 1 : Google OR-Tools uses

    # iCollect permits the routing and navigation of vehicles with different settings for "DepartureTimeFromMinutes".
    # In order to achieve this, activate both the Distance Matrix and the Directions APIs and run iCollect with the desired settings.
    # Then, disable the API of your chooding and run iCollect again with a different setting.
    #
    start_measuring_runtime = time.time()

    main(BinCapacity=1100, FillLevelThreshold=0, DepartureTimeFromMinutes=0, ArrivalTimeFromMinutes=480)

    print(time.time() - start_measuring_runtime)
