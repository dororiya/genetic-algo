import random
import Methuselah
import tkinter as tk
import numpy as np
import save_methuselah as sm
import matplotlib.pyplot as plt


SIZE_ROW = 40
SIZE_COL = 40

MY_METHUSELAH_LIST = sm.SaveMethuselah('save_methuselah.yaml')
CURRENT_METHUSELAH_ID = -1  # first in my list is id = 0


# create starting population with size I chose in the middle of the board
def create_starting_population(num_starting_population):
    population = []
    for _ in range(num_starting_population):
        num_cells = random.randint(3, 10)
        live_cells = set()   # makes sure there are no repetitions
        while len(live_cells) < num_cells:  # check has required cell
            random_row = random.randint(SIZE_ROW // 2 - 3, SIZE_ROW // 2 + 3)
            random_col = random.randint(SIZE_COL // 2 - 3, SIZE_COL // 2 + 3)
            live_cells.add((random_row, random_col))
        population.append(list(live_cells))     # methuselah get list
    first_population = [Methuselah.Configuration(config) for config in population]
    return first_population


# chose parents to the next generation
def next_parents(population, probability_list):
    total_parents = 45  # number of parents I want to select
    parents = set()  # To store unique pairs of parents

    while len(parents) < total_parents:
        # Select two parents at random with probability weighting
        father, mother = random.choices(population, weights=probability_list, k=2)

        # Ensure that the pair is not already selected (father != mother)
        if father != mother:
            # Sort the pair to avoid (father, mother) and (mother, father) being counted as different
            parents.add(tuple(sorted((father, mother))))

    # Convert set back to list of pairs
    return [list(pair) for pair in parents]


# chose survivors (the best individual always survive)
def next_survivors(population, probability_list):
    total_survivors = 5  # Number of survivors you want to select
    # Find the individual with the maximum fitness
    max_fitness_index = probability_list.index(max(probability_list))  # Index of the highest fitness
    max_fitness_individual = population[max_fitness_index]  # The individual with max fitness
    survivors = [max_fitness_individual]    # max fitness survive and 3 more other
    while len(survivors) < total_survivors:
        survivor = random.choices(population, weights=probability_list, k=1)[0]
        if survivor not in survivors:
            survivors.append(survivor)
    return survivors


# add new cell to the list of cells
def add_to_cell(lst):
    while True:
        random_row = random.randint(SIZE_ROW // 2 - 4, SIZE_ROW // 2 + 4)
        random_col = random.randint(SIZE_COL // 2 - 4, SIZE_COL // 2 + 4)
        cell = (random_row, random_col)
        if cell not in lst:
            lst.append(cell)
            break


# chose option of mutation and execute
def mutation(child):
    to_do = random.choice(['add', 'remove', 'switch'])
    if to_do == 'remove' and child:
        # Remove a random cell if the child has more than one cell
        del child[random.choice(range(len(child)))]
    elif to_do == 'add':
        # Add a random cell
        add_to_cell(child)
    elif to_do == 'switch' and child:
        # Replace a random cell with a new one
        cell_to_change = random.choice(range(len(child)))
        new_cell = (random.randint(SIZE_ROW // 2 - 4, SIZE_ROW // 2 + 4), random.randint(SIZE_COL // 2 - 4, SIZE_COL // 2 + 4))
        while new_cell in child:
            new_cell = (random.randint(SIZE_ROW // 2 - 4, SIZE_ROW // 2 + 4), random.randint(SIZE_COL // 2 - 4, SIZE_COL // 2 + 4))
        child[cell_to_change] = new_cell

    # Ensure the child is not empty after mutation
    if not child:
        # Add a random cell if child becomes empty
        add_to_cell(child)


# get parents and create child from them
def create_children(parents):
    fit_father = parents[0].get_fitness()
    fit_mother = parents[1].get_fitness()
    total_fit = fit_father + fit_mother
    father_list_of_live_cell = parents[0].get_list_of_live_cell()
    mother_list_of_live_cell = parents[1].get_list_of_live_cell()
    father_cell = len(father_list_of_live_cell)
    mother_cell = len(mother_list_of_live_cell)
    average_len = (father_cell + mother_cell) // 2
    len_option = [father_cell, mother_cell, average_len]
    # chose one option of number of cells from len_option
    if total_fit == 0:
        num_cell = random.choices(len_option, weights=[1/3, 1/3, 1/3], k=1)[0]
    else:
        num_cell = random.choices(len_option, weights=[fit_father / (2 * total_fit), fit_mother / (2 * total_fit), 0.5], k=1)[0]

    add_from_father = min(1, int((fit_father / total_fit)) * num_cell)

    if random.random() < 0.5:
        child = father_list_of_live_cell[: add_from_father]
    else:
        child = father_list_of_live_cell[-add_from_father:]
    add_to_child = num_cell - len(child)
    not_in_child = [item for item in mother_list_of_live_cell if item not in child]
    if len(not_in_child) >= add_to_child:   # add from mother
        child.extend(not_in_child[:add_to_child])
    else:   # add from father
        child.extend(not_in_child)
        add_to_child -= len(not_in_child)
        not_in_child_father = [item for item in father_list_of_live_cell if item not in child]
        child.extend(not_in_child_father[:add_to_child])
    mutation_probability = 0.3  # chosen probability for mutation
    if random.random() < mutation_probability:  # doing mutation in chosen probability
        mutation(child)
    return Methuselah.Configuration(child)


# create next generation by choose survivors and parents and children
def next_generation(population):
    fitness_list = [config.get_fitness() for config in population]
    sum_fitness = sum(fitness_list)
    try:
        probability_list = [fitness / sum_fitness for fitness in fitness_list]
    except ZeroDivisionError:
        probability_list = [1 / len(fitness_list)] * len(fitness_list)
    my_parents = next_parents(population, probability_list)
    my_survivors = next_survivors(population, probability_list)
    my_children = [create_children(parents) for parents in my_parents]
    return my_survivors + my_children


# return label text
def get_text_methuselah(time_to_converge, max_size):
    text = ("this methuselah converge after {} iteration and his max size is "
            "{}").format(time_to_converge, max_size)
    return text


# update the canvas update the canvas to the next methuselah in the yamal file
def next_methuselah(canvas, cell_size, label):
    global CURRENT_METHUSELAH_ID, MY_METHUSELAH_LIST
    CURRENT_METHUSELAH_ID = (CURRENT_METHUSELAH_ID + 1) % MY_METHUSELAH_LIST.count_methuselah()
    current_met = MY_METHUSELAH_LIST.get_methuselah(CURRENT_METHUSELAH_ID)
    if current_met is not None:  # else do nothing
        # Clear the canvas
        canvas.delete("all")
        lst, max_size, time_to_converge = current_met
        label.config(text=get_text_methuselah(time_to_converge, max_size))
        # Loop through the space and draw each cell
        for row in range(SIZE_ROW):
            for col in range(SIZE_COL):
                # If the cell is alive (1), color it green, otherwise white
                cell = (row, col)
                color = "green" if cell in lst else "white"
                canvas.create_rectangle(col * cell_size, row * cell_size,
                                        (col + 1) * cell_size, (row + 1) * cell_size,
                                        fill=color, outline="black")


# return the list of live cell in the next iteration of convey game
def convey_game_next_iteration(live_cell):
    space = np.zeros((SIZE_ROW, SIZE_COL))
    rows, cols = zip(*live_cell)
    space[rows, cols] = 1
    list_live_cell = []
    for row in range(SIZE_ROW):
        for col in range(SIZE_COL):
            neighbours = Methuselah.num_of_neighbours(row, col, space)
            if neighbours == 2 and space[row, col] == 1:
                list_live_cell.append((row, col))
            elif neighbours == 3:
                list_live_cell.append((row, col))
    return list_live_cell


# update the canvas to the list of live cell the function get as input
def update_canvas(live_cell, canvas, cell_size):
    # Clear the canvas
    canvas.delete("all")

    # Loop through the space and draw each cell
    for row in range(SIZE_ROW):
        for col in range(SIZE_COL):
            # If the cell is alive (1), color it green, otherwise white
            cell = (row, col)
            color = "green" if cell in live_cell else "white"
            canvas.create_rectangle(col * cell_size, row * cell_size,
                                    (col + 1) * cell_size, (row + 1) * cell_size,
                                    fill=color, outline="black")


# run the methuselah on the canvas
def run_methuselah(root, canvas, time_to_converge, iteration, live_cell, cell_size):
    num_of_ml = 5  # Delay in milliseconds
    iteration += 1

    # Update live cells for the next iteration
    live_cell = convey_game_next_iteration(live_cell)

    # Update the canvas with the new state
    update_canvas(live_cell, canvas, cell_size)

    # Continue if the number of iterations has not reached convergence
    if iteration < time_to_converge:
        root.after(num_of_ml, run_methuselah, root, canvas, time_to_converge, iteration, live_cell, cell_size)
    else:
        print("Simulation Complete.")


# return the best methuselah as 3-tuple
def get_curr_val(best_methuselah):
    lst = best_methuselah.get_list_of_live_cell()
    time_to_converge = best_methuselah.get_time_to_converge()
    max_size = best_methuselah.get_max_size()
    return lst, time_to_converge, max_size


# create the tkinter
def display(best_methuselah):
    num_of_ml = 5

    lst, time_to_converge, max_size = get_curr_val(best_methuselah)

    # Create a tkinter window
    root = tk.Tk()
    root.title("Conway's Game of Life")
    label = tk.Label(root, text=get_text_methuselah(time_to_converge, max_size), font="Ariel")
    label.pack()

    # Set the size of each cell in the grid
    cell_size = 20

    # Create a canvas to draw the grid
    canvas = tk.Canvas(root, width=SIZE_COL * cell_size, height=SIZE_ROW * cell_size)
    canvas.pack()

    # Loop through the space and draw each cell
    for row in range(SIZE_ROW):
        for col in range(SIZE_COL):
            # If the cell is alive (1), color it green, otherwise white
            cell = (row, col)
            color = "green" if cell in best_methuselah.get_list_of_live_cell() else "white"
            canvas.create_rectangle(col * cell_size, row * cell_size,
                                    (col + 1) * cell_size, (row + 1) * cell_size,
                                    fill=color, outline="black")
    switch_button = tk.Button(text="switch methuselah", command=lambda: next_methuselah(canvas,
                                                                                        cell_size, label), bg="red")
    switch_button.place(x=10, y=10)

    save_button = tk.Button(text="save methuselah",
                            command=lambda: MY_METHUSELAH_LIST.add_methuselah_to_config(lst,
                                                                                        time_to_converge, max_size),
                            bg="red")
    save_button.place(x=10, y=100)
    start = tk.Button(
        text="Start Methuselah",
        command=lambda: root.after(num_of_ml, run_methuselah, root, canvas,
                                   best_methuselah.get_time_to_converge(), 0,
                                   best_methuselah.get_list_of_live_cell(),
                                   cell_size),
        bg="red"
    )
    start.pack()
    # Run the tkinter event loop
    root.mainloop()


# create plot of the data of the genetic algorithm
def plot_the_data(mean_fitness, max_fitness, num_of_generations):
    # Plotting the mean and max fitness over generations
    plt.figure(figsize=(10, 6))

    # Plot mean fitness
    plt.plot(range(1, num_of_generations + 1), mean_fitness, label='Mean Fitness', color='blue', linestyle='-',
             marker='o')

    # Plot max fitness
    plt.plot(range(1, num_of_generations + 1), max_fitness, label='Max Fitness', color='red', linestyle='-', marker='x')

    # Adding titles and labels
    plt.title('Fitness Evolution Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    # Show the plot
    plt.show()


# main of this file
def main():
    # Create the starting population
    my_population = create_starting_population(50)
    num_of_generations = 20

    matrix_of_fit = np.zeros((num_of_generations, len(my_population)))
    mean_fitness = []  # To store the mean fitness per generation
    max_fitness = []  # To store the max fitness per generation

    for generation in range(num_of_generations):
        # Print the population of live cells in this generation
        print(f"Generation {generation + 1}:")
        for i, individual in enumerate(my_population):
            curr_fit = individual.get_fitness()
            matrix_of_fit[generation, i] = curr_fit
            print(f"Individual {i + 1}: {individual.get_list_of_live_cell()} fitness:{curr_fit}")

        # Compute mean and max fitness for the current generation
        mean_fitness.append(np.mean(matrix_of_fit[generation]))  # Mean fitness
        max_fitness.append(np.max(matrix_of_fit[generation]))  # Max fitness

        # Generate the next generation
        my_population = next_generation(my_population)
        print("-" * 50)  # Print a separator for clarity
    best_methuselah = sorted(my_population, key=lambda x: x.get_fitness(), reverse=True)[0]
    plot_the_data(mean_fitness, max_fitness, num_of_generations)
    display(best_methuselah)


if __name__ == "__main__":
    main()











