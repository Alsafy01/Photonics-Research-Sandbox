
"""
    Raw explanation of the main parts of Genetics Algorithm,
    code is very general and only buts the main functions in its place
    no external packages been used other than "numpy" been used for simplification
    and explore the capabilities of taking this implementation to a lower level
    """

import numpy as np

# Define parameters
population_size = 100
num_generations = 50
mutation_rate = 0.1
crossover_rate = 0.8
num_parameters = 10  # Assuming there are 10 parameters in the design


# Encoding process
def initialize_population(population_size, num_parameters):
    # Initialize population within a certain range
    population = np.random.uniform(low=-1, high=1, size=(population_size, num_parameters))
    return population


# Placeholder for your actual objective function
def evaluate_solution(solution):
    # Replace this with your actual objective function
    # For now, return a random fitness score
    return np.random.rand()


# Selection process
def select_parents(population, fitness_scores):
    # Select parents based on fitness scores using tournament selection
    tournament_size = 5
    parents_indices = []
    for _ in range(len(population) // 2):
        tournament = np.random.choice(len(population), size=tournament_size, replace=False)
        winner = tournament[np.argmax(fitness_scores[tournament])]
        parents_indices.append(winner)
    parents = population[parents_indices]
    return parents


# Crossover operation
def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent1))  # Ensure at least one parameter is selected
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    else:
        child1, child2 = parent1.copy(), parent2.copy()  # If no crossover, offspring are identical to parents
    return child1, child2


# Mutation operation
def mutate(solution):
    for i in range(len(solution)):
        if np.random.rand() < mutation_rate:
            solution[i] += np.random.normal(scale=0.1)  # Add Gaussian noise with standard deviation 0.1
            solution[i] = np.clip(solution[i], -1, 1)  # Clip values to stay within a certain range
    return solution


# DNN-based evolution
def evolve(parents):
    # Use DNN model to replace crossover and mutation
    # Placeholder: Implement this based on your DNN model
    # Forward design process
    # Obtain device FOM and network weight parameters
    # Inverse design process
    # Generate offspring device designs correlated but different from parents
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            child1, child2 = crossover(parents[i], parents[i + 1])
            child1 = mutate(child1)
            child2 = mutate(child2)
            offspring.extend([child1, child2])
        else:
            # If the number of parents is odd, the last parent becomes the only offspring
            offspring.append(mutate(parents[i]))
    return offspring


# Genetic algorithm loop
population = initialize_population(population_size, num_parameters)
for generation in range(num_generations):
    # Evaluate the fitness of each solution in the population
    fitness_scores = np.array([evaluate_solution(solution) for solution in population])
    print("Generation:", generation, "  Best Fitness:", np.max(fitness_scores))

    # Select parents for reproduction
    parents = select_parents(population, fitness_scores)

    # Ensure the number of parents is even
    if len(parents) % 2 != 0:
        parents = parents[:-1]

    # DNN-based evolution (crossover and mutation)
    offspring = evolve(parents)

    # Ensure the population size remains constant
    while len(offspring) < population_size:
        offspring.append(mutate(parents[0].copy()))  # Add mutated copies of parents if needed

    # Replace the old population with the offspring
    population = np.array(offspring)

# Select the best solution from the final population
fitness_scores = np.array([evaluate_solution(solution) for solution in population])
best_solution_index = np.argmax(fitness_scores)
best_solution = population[best_solution_index]

print("\nBest solution after", num_generations, "generations:", best_solution)
print("Best fitness:", fitness_scores[best_solution_index])
