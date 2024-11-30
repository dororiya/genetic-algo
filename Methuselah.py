import numpy as np

SIZE_ROW = 40
SIZE_COL = 40
ITERATION_TO_CONVERGE = 450


# define configuration
class Configuration(object):
    def __init__(self, list_of_live_cell=None):
        self.list_of_live_cell = list_of_live_cell
        self.space = np.zeros((SIZE_ROW, SIZE_COL))
        if self.list_of_live_cell:   # not empty list
            rows, cols = zip(*self.list_of_live_cell)  # Unzips the list into rows and cols
            self.space[rows, cols] = 1  # method that fit to numpy array to fill the matrix with 1 in the place I want

            # time_to_converge =0 mean not converge
            self.max_size, self.time_to_converge = fitness_function(self, ITERATION_TO_CONVERGE)
            self.list_of_live_cell = list_of_live_cell
            self.space = np.zeros((SIZE_ROW, SIZE_COL))
            self.space[rows, cols] = 1

            # if not converge is not 'Methuselah', take less than 50 iteration (Martin Gardner defined methuselahs)
            if self.time_to_converge < 50:  # Check number of live cells
                self.fitness = (self.max_size / len(self.list_of_live_cell)) * (self.time_to_converge / 100)
            else:  # For configurations with 50 or more live cells
                self.fitness = self.max_size / len(self.list_of_live_cell)
        else:
            self.max_size, self.time_to_converge = 0, 0
            self.fitness = 0

    def get_list_of_live_cell(self):
        return self.list_of_live_cell.copy()

    def get_space(self):
        return self.space.copy()

    def get_fitness(self):
        return self.fitness

    def get_time_to_converge(self):
        return self.time_to_converge

    def get_max_size(self):
        return self.max_size

    def set_list_of_live_cell(self, new_list_of_live_cell):
        self.list_of_live_cell = new_list_of_live_cell
        self.space = np.zeros((SIZE_ROW, SIZE_COL))
        if new_list_of_live_cell:  # Only unpack if the list is not empty
            rows, cols = zip(*self.list_of_live_cell)
            self.space[rows, cols] = 1

    # little then compare the fitness
    def __lt__(self, other):
        return self.get_fitness() < other.get_fitness()

    # we define __hash__ and __eq__  to work correctly as a dictionary key or in a set
    # find as a good way to compare the configuration
    def __hash__(self):
        # Hash based on the list of live cells (convert it to a tuple since lists are not hashable)
        return hash(tuple(self.list_of_live_cell))

    def __eq__(self, other):
        # Equality check based on the list of live cells
        return isinstance(other, Configuration) and sorted(self.list_of_live_cell) == sorted(other.list_of_live_cell)


# fitness check if the configuration converge and his max size
def fitness_function(con, iteration):
    list_of_live_cells = con.get_list_of_live_cell()
    max_size = len(list_of_live_cells)
    previous_configuration = set()  # makes sure there are no repetitions
    previous_configuration.add(tuple(sorted(con.get_list_of_live_cell())))
    for i in range(iteration):
        converge = convey_game_next_iteration(con, previous_configuration)
        live_cells_num = len(con.get_list_of_live_cell())
        if max_size < live_cells_num:
            max_size = live_cells_num
        if converge:
            return max_size, i
    return max_size, 0  # if not converge fitness be zero(not methuselah)


# check neighbours of cell
def num_of_neighbours(row, col, space):
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    count = 0
    for ar, ac in neighbours:
        nr, nc = (row + ar) % SIZE_ROW, (col + ac) % SIZE_COL
        if space[nr, nc] == 1:
            count += 1
    return count


# return true if converge else false
def convey_game_next_iteration(con, previous_configuration):
    space = con.get_space()
    list_live_cell = []     # build every call new list and fill it's with convey rule
    for row in range(SIZE_ROW):
        for col in range(SIZE_COL):
            neighbours = num_of_neighbours(row, col, space)
            if neighbours == 2 and space[row, col] == 1:    # live cell and 2 neighbours(keep a live)
                list_live_cell.append((row, col))
            elif neighbours == 3:
                list_live_cell.append((row, col))
    con.set_list_of_live_cell(list_live_cell)
    this_configur = tuple(sorted(con.get_list_of_live_cell()))  # hashable representation
    if this_configur in previous_configuration:
        return True
    previous_configuration.add(this_configur)
    return False


