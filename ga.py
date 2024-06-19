import numpy as np
import Game2048
import tqdm as tqdm

# Should be random, but ill come back to that
tile1 = (0, 0)
tile2 = (0, 1)
num_individuals = 5
dim = 10  # 10 moves
generations = 10
lamda = 1  # offspring coefficient
mutation_rate = 0.01

def fitness(individual):
    game = Game2048.Game2048()
    game.reset(tile1, tile2)
    moves = individual.reshape((int(dim/2), 2))
    print(game, "\n")
    for move in moves:
        if game.move(str(move[0]) + str(move[1])) == -1:
            pass
        print(game, '\n')
    print(game)
    return game.score

# individuals = np.zeros((num_individuals, dim))
# # initialize population
# for i in range(num_individuals):
#     individuals[i] = np.random.randint(0, 2, dim)

# best_individual = None
# best_fitness = -1

# for i in tqdm.tqdm(range(generations)):
#     new_offspring = np.zeros((num_individuals * lamda, dim))
#     for j in range(num_individuals * lamda):
#         # select parents
#         parents = np.random.choice(num_individuals, 2, replace=False)
#         # crossover
#         crossover_point = np.random.randint(1, dim)
#         offspring = np.concatenate((individuals[parents[0]][:crossover_point], individuals[parents[1]][crossover_point:]))
#         # mutate
#         mutation_mask = np.random.rand(dim) < mutation_rate
#         offspring[mutation_mask] = 1 - offspring[mutation_mask]
#         new_offspring[j] = offspring

#     temp_individuals = np.vstack((individuals, new_offspring))
#     fitnesses = np.zeros(len(temp_individuals))
#     for i, individual in enumerate(temp_individuals):
#         fitnesses[i] = fitness(individual)
#     i = 0
#     for individual in temp_individuals:
#         fitnesses[i] = fitness(individual)
#         print(individual)
#         print(fitness(individual))
    
#     print("Fitnesses:", fitnesses)        
    
    
    
#     # Update best individual and fitness
#     max_fitness_idx = np.argmax(fitnesses)
#     if fitnesses[max_fitness_idx] > best_fitness:
#         best_individual = temp_individuals[max_fitness_idx]
#         best_fitness = fitnesses[max_fitness_idx]
        
#     print(individuals)
#     print(fitnesses)
#     print(f"Best Fitness = {best_fitness}")
#     print(f"Best Individual = {best_individual}")
    
#     # Sort individuals in descending order based on fitness
#     sorted_indices = np.argsort(fitnesses)[::-1]
#     individuals = temp_individuals[sorted_indices]
#     individuals = individuals[:num_individuals]
    
#     print("\n\n\n")

# print("Best Individual:")
# print(best_individual)

# print("Best Fitness:")
# print(best_fitness)

# print("Best Individual's Game:")
# print(fitness(best_individual))

# print("Fitnesses:")
# print(fitnesses)

individual = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 1])

print(fitness(individual), "\n\n")
print(fitness(individual))