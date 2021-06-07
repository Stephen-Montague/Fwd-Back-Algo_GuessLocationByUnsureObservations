import numpy as np

# Prints solution to the console.
def print_world(grid, robot):
    # Print map of grid with robot location.
    print("\nRobot Location Report")
    print("Error Rate:", robot.error_rate)
    print("=================")
    print("\nGrid Map:")
    grid_space_name = 1
    robot_space_name = None
    robot_location_index = 0
    for index, space in enumerate(grid.layout):
        if index % grid.width == 0:
            print('')
        if space == 0:
            if robot.location == robot_location_index:
                print("\t\t @_@-", end='')
                print(grid_space_name, end='')
                robot_space_name = grid_space_name
            else:
                print("\t\t\t", grid_space_name, end='')
            grid_space_name += 1
            robot_location_index += 1
        else:
            print('\t\t\tXXX', end='')
    print("\n\nRobot ends on space:", robot_space_name)
    print("Forward Probability:", "{:.3f}".format(robot.fwd_location_distributions[-1][robot.location]))

    # Print Forward and FWD-Backward distributions
    print("=================")
    print("\nForward distributions:\t\tFWD-Backward Distributions:")
    time_step_count = len(robot.sensor_readings)
    for t in range(time_step_count):
        print(f"\nStep {t+1}:")
        for i in range(grid.open_location_count):
            print(f"{robot.fwd_location_distributions[t][i]:.16f}\t\t\t"
                  f" {robot.fwd_back_location_distributions[t][i]:.16f}", end='')
            if robot.fwd_back_location_distributions[t][i] == max(robot.fwd_back_location_distributions[t]):
                print("-@_@ on space:", i+1)
            else:
                print()
    print("\n^^^ Forward ^^^\t\t\t\t^^^ FWD-Backward ^^^")
    print("\nRobot Error Rate:", robot.error_rate)
    print("End Report")
    print("=================")
    return

class Grid:
    def __init__(self, layout=None, correct_readings=None):
        if layout is None:
            self.layout = [0, 0, 0, 0,  # Zero is open, one is a wall.
                           1, 1, 0, 0,
                           1, 0, 0, 0,
                           0, 0, 1, 0]
            self.correct_readings = ["NSW", "NS", "N", "NE",
                                     "XXX", "XXX", "W", "E",
                                     "XXX", "NW", "S", "E",
                                     "NSW", "SE", "XXX", "SEW"]
        else:
            self.layout = layout
            self.correct_readings = correct_readings
        self.open_location_count = self.layout.count(0)
        self.width = np.sqrt(len(self.layout))


# noinspection PyTypeChecker
class Robot:
    def __init__(self, grid, error_rate=0.2, sensor_readings=None):
        self.error_rate = error_rate
        if sensor_readings is None:
            self.sensor_readings = ["NSW", "SE", "NW", "S", "E", "E"]
        else:
            self.sensor_readings = sensor_readings
        self.time_step_count = len(self.sensor_readings) + 1  # Add 1 to count prior state 0.
        self.possible_locations = []  # Useful to iterate over the grid skipping walls.
        for index, space in enumerate(grid.layout):
            if space == 0:
                self.possible_locations.append(index)
        self.prob_of_observations = self.set_prob_of_observations(grid)
        self.prob_of_transitions = self.set_prob_of_transition(grid)
        self.fwd_location_distributions = self.set_fwd_location_distributions(grid)
        self.fwd_back_location_distributions = self.set_fwd_back_location_distributions()
        self.location = self.guess_location()

    # Returns a list, each element a distribution of the robot's location in the grid after n observations.
    def set_fwd_location_distributions(self, grid):
        fwd_transition_matrix = np.transpose(self.prob_of_transitions.copy())
        priors = [[1 / grid.open_location_count] * grid.open_location_count]  # Index 0 has time-step 0 prior.
        time_step = 1
        distribution = []
        while time_step < self.time_step_count:
            observation_matrix = self.set_observation_matrix(self.prob_of_observations[time_step])
            prior = priors[time_step - 1]
            sub_product = observation_matrix @ fwd_transition_matrix @ prior
            alpha = 1 / sum(sub_product)
            final_product = alpha * sub_product
            distribution.append(final_product)
            time_step += 1
            priors.append(final_product)
        return distribution

    def set_fwd_back_location_distributions(self):
        back_transition_matrix = np.array(self.prob_of_transitions)
        back_msgs = [self.fwd_location_distributions[-1]]  # b == normalized (vector of ones * last_fwd) == last_fwd.
        time = 1  # Time 0 is future, time 1 is last step, increment back to first step.
        max_time = self.time_step_count - 1  # Allow one iteration per time-step, minus last time-step and prior.
        observation_index = len(self.sensor_readings)  # Decrement from last to first observation, skip prior- index 0.
        fwd_index = len(self.sensor_readings) - 2  # Decrement from 2nd-to-last fwd_distribution to first, at index 0.
        distributions = [back_msgs[0]]  # Set end distribution, append further back distributions, and reverse.
        while time < max_time:
            observation_matrix = self.set_observation_matrix(self.prob_of_observations[observation_index])
            back_msg = back_transition_matrix @ observation_matrix @ back_msgs[time - 1]
            fwd_back_product = self.fwd_location_distributions[fwd_index] * back_msg
            alpha = 1/sum(fwd_back_product)
            final_product = alpha * fwd_back_product
            distributions.append(final_product)
            time += 1
            observation_index -= 1
            fwd_index -= 1
            if time < max_time:
                back_msgs.append(final_product)
        return list(reversed(distributions))

    # Sets the independent probability of all states given the sensor readings.
    # Returns a 2d numpy array: an array of time-steps, each with an array of location probabilities.
    def set_prob_of_observations(self, grid):
        prob_of_observations = [np.ones(grid.open_location_count)]  # Probability of no observation in prior state is 1.
        sensor_readings = self.sensor_readings.copy()
        while len(sensor_readings) > 0:
            current_readings = sensor_readings.pop(0)
            probabilities = np.zeros(grid.open_location_count)
            e = self.error_rate
            for location_i, location in enumerate(self.possible_locations):
                difference_count = 0
                for reading in current_readings:
                    if reading not in grid.correct_readings[location]:
                        difference_count += 1
                for correct_reading in grid.correct_readings[location]:
                    if correct_reading not in current_readings:
                        difference_count += 1
                probabilities[location_i] = ((1 - e) ** (4 - difference_count)) * e ** difference_count
            prob_of_observations.append(probabilities)
        return prob_of_observations

    # noinspection PyMethodMayBeStatic
    def set_observation_matrix(self, observations):
        size = len(observations)
        matrix = np.zeros((size, size))
        np.fill_diagonal(matrix, observations.copy())
        return matrix

    # Returns numpy 2d array of all origin to destination probabilities
    def set_prob_of_transition(self, grid):
        possible_origins = self.possible_locations
        possible_destinations = self.possible_locations
        transition_matrix = np.zeros((grid.open_location_count, grid.open_location_count))
        for i, origin in enumerate(possible_origins):
            legal_move_count = self.set_legal_move_count(grid, origin)
            for j, destination in enumerate(possible_destinations):
                if self.has_legal_move(grid, origin, destination):
                    transition_matrix[i, j] = 1 / legal_move_count  # Sets P(move) into given destination
        return transition_matrix

    def set_legal_move_count(self, grid, origin):
        legal_move_count = 0
        possible_moves = [left, right] = [origin - 1, origin + 1]
        possible_moves.extend([origin - grid.width, origin + grid.width])  # North, south
        left_side = 0
        right_side = grid.width - 1
        relative_location = origin % grid.width  # Used to check left or right side relative to grid, see below.
        for move in possible_moves:
            if move in self.possible_locations:
                if relative_location != left_side and relative_location != right_side:
                    legal_move_count += 1
                elif relative_location == left_side and move != left:
                    legal_move_count += 1
                elif relative_location == right_side and move != right:
                    legal_move_count += 1
        return legal_move_count

    # noinspection PyMethodMayBeStatic
    def has_legal_move(self, grid, origin, destination):
        possible_moves = [left, right] = [origin - 1, origin + 1]
        possible_moves.extend([origin - grid.width, origin + grid.width])  # North, south
        left_side = 0
        right_side = grid.width - 1
        relative_location = origin % grid.width  # Relative to sides of grid
        if destination in possible_moves:
            if relative_location != left_side and relative_location != right_side:
                return True
            elif relative_location == left_side and destination != left:
                return True
            elif relative_location == right_side and destination != right:
                return True
        return False

    def guess_location(self):
        max_index = 0
        max_value = 0
        for index, value in enumerate(self.fwd_location_distributions[-1]):
            if value > max_value:
                max_value = value
                max_index = index
        return max_index


# Test - Lecture Grid
test_layout = [0, 0, 1,
               1, 0, 0,
               1, 0, 1]
test_correct_readings = ["NSW", "NE", "XXX",
                         "XXX", "W", "NSE",
                         "XXX", "SEW", "XXX"]
test_sensor_readings = ["NE", "NSW"]

# Main:
# Grid and Robot given no other parameters runs default assignment.
# Test code above can be used as parameters for the lecture example.
# Default error rate is 0.2, accessible via parameter, e.g.: Robot(error_rate=0).

my_grid = Grid()
my_robot = Robot(my_grid)
print_world(my_grid, my_robot)
