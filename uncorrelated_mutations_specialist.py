"""Final file for the assignment 1 of the course Evolutionary Computing course"""

import os
import pickle
import time
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt

from player_controller import player_controller
from evoman.environment import Environment

from utils import verify_solution

# ---- Initialization
# Set headless
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
# Set experiment name
experiment_name = 'final_specialist_assignment'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
# Set number of hidden neurons
n_hidden_neurons = 10


def initialize(mu, n, limits, mag_sigma, mutation_strategy):
    """Goal:    
        The function initializes the algorithm.
    -------------------------------------------------------
    Input:
        mu:
            Parent population size, int
        n:
            Number of variables to optimize, int
        limits:
            Lower and Upper Boundary of x_0, list
        mag_sigma:
            Magnitude of sigma, float
        mutation_strategy:
            Mutation strategy, string
    -------------------------------------------------------
    Output:
        sigma:
            Sigma values(s), float/np.array
        x_0:
            Initial parent population,
                n x mu np.array"""

    # Generate random parent population
    x_0 = np.random.uniform(limits[0], limits[1], (mu, n))

    # Generate sigma(s)
    if mutation_strategy == "self-adaptive uncorrelated 1 stepsize":
        sigma = np.random.uniform(0, mag_sigma, size=mu)
    elif mutation_strategy == "self-adaptive uncorrelated n stepsizes":
        sigma = np.random.uniform(0, mag_sigma, size=(mu, n))
    else:
        sigma = mag_sigma

    return x_0, sigma


# def calculate_amount_of_similar_solutions(pop) -> int:
#     similar_solutions = 0
#     for i in range(len(pop)):
#         for j in range(i + 1, len(pop)):
#             euclidean_distance = np.linalg.norm(pop[i, :] - pop[j, :])
#             # print(f"Euclidean distance between solution {i} and solution {j} = {euclidean_distance}")
#             if euclidean_distance < 30:
#                 similar_solutions += 1
#     return similar_solutions

def parent_selection(pop, pop_sigma, pop_fit, n_parents, parent_selection_strategy, smoothing=1):
    """Goal:
        Perform parent selection.
    ---------------------------------------------------------------------------------
    Input:
        pop:
            Population, np array
        pop_sigma:
            Population sigma, np array/float
        pop_fit:
            Population fitness, np array
        n_parents:
            Number of parents, int
        parent_selection_strategy:
            Parent selection strategy, string
            Rank or Fitness Proportional
        smoothing:
            Smoothing parameter, float
    ---------------------------------------------------------------------------------
    Output:
        population:
            Selected parents, np array
        sigma_pop:
            Selected parents sigma, list/float"""

    # ---- Adapt fitness
    fitness = pop_fit + smoothing - np.min(pop_fit)

    # ---- Set selection probabilities
    # if parent_selection_strategy == "Rank":  # Rank Selection
    #     # Sort by fitness
    #     sorted_indices = np.argsort(fitness)
    #     pop = pop[sorted_indices]
    #     if type(pop_sigma) in [float, int]:
    #         pop_sigma = pop_sigma
    #     else:
    #         pop_sigma = pop_sigma[sorted_indices]
    #     # Give rank to each individual based on fitness
    #     rank = np.arange(len(pop) - 1, -1, -1)  # unity, i.e., it is a function of the population size
    #     # Generate prop vector based on rank
    #     P_exp_rank = (1 - np.exp(-rank))
    #     # Ensure that prop vector sums to 1
    #     P_exp_rank = P_exp_rank / np.sum(P_exp_rank)
    #     # Set selection probabilities
    #     selection_probabilities = P_exp_rank
    # elif parent_selection_strategy == "Fitness Proportional":  # Fitness Proportional Selection
    # Fitness proportional selection probability
    fps = fitness / np.sum(fitness)
    selection_probabilities = fps
    # else:
    #     raise ValueError("Unknown parent selection strategy.")

    # Make a random selection of indices
    parent_indices = np.random.choice(np.arange(0, pop.shape[0]), (n_parents, 2), p=selection_probabilities)

    # # ensure that no parents indices are the same
    # while np.any(parent_indices[:, 0] == parent_indices[:, 1]):
    #     # get indices where the parents are the same
    #     same_parents_indices = np.where(parent_indices[:, 0] == parent_indices[:, 1])[0]
    #     # print("same parents")
    #     # print(same_parents_indices)
    #     # Choose new parents for the same parents indices
    #     parent_indices[same_parents_indices, :] = np.random.choice(np.arange(0, pop.shape[0]),
    #                                                                (len(same_parents_indices), 2),
    #                                                                p=selection_probabilities)

    # Set population
    population = pop[parent_indices, :]

    # Population sigma
    if type(pop_sigma) in [float, int]:
        sigma_pop = pop_sigma
    else:
        # Initialize
        sigma_pop = [[], []]
        # Get sigma
        for parent_index in parent_indices:
            for parent, baby in enumerate(parent_index):
                sigma_pop[parent].append(pop_sigma[baby])

    return population, sigma_pop


def crossover(parents, parents_sigma, mutation_strategy, limits, alfa=0.5, sigma_min=0.0001):
    """Goal:
        Perform crossover.
    ---------------------------------------------------------------------------------
    Input:
        parents:
            Parents, np array
        parents_sigma:
            Parents sigma, np array
        mutation_strategy:
            Mutation strategy, string
        limits:
            Lower and Upper Boundary of x_0, list
        alfa:
            Crossover parameter for x, float
        sigma_min:
            Minimum sigma, float
    ---------------------------------------------------------------------------------
    Output:
        offspring:
            Offspring, np array
        off_sigma:
            Offspring sigma, float/list"""

    # Split into parents
    parent1, parent2 = np.hsplit(parents, 2)
    if mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
        parent1_sigma, parent2_sigma = parents_sigma[0], parents_sigma[1]
    elif mutation_strategy in ["uniform", "non-uniform"]:
        parent1_sigma, parent2_sigma = parents_sigma, parents_sigma
    else:
        raise ValueError("Unknown mutation strategy.")

    # ---- x --> Blend Recombination
    # Sample random number between 0 and 1
    u = np.random.uniform(0, 1)
    # Get gamma
    mult = (1 - 2 * alfa) * u - alfa
    # Perform crossover
    offspring = mult * parent1 + (1 - mult) * parent2  # Child 1
    offspring2 = (1 - mult) * parent1 + mult * parent2
    offspring = np.concatenate((offspring, offspring2), axis=0)
    # Adjust for boundaries
    offspring[offspring < limits[0]] = limits[0]
    offspring[offspring > limits[1]] = limits[1]
    # ---- sigma --> Blend Recombination
    if mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
        # Perform crossover
        off_sigma = [mult * np.array(parent1_sigma[i]) + (1 - mult) * np.array(parent2_sigma[i]) for i in
                     range(0, len(parent1_sigma))]  # Child 1
        off_sigma += [(1 - mult) * np.array(parent1_sigma[i]) + mult * np.array(parent2_sigma[i]) for i in
                      range(0, len(parent1_sigma))]  # Child 2 --> both childs

        # Adjust for boundaries
        for i in range(len(off_sigma)):
            off_sigma[i] = np.where(off_sigma[i] < sigma_min, sigma_min, off_sigma[i])
    else:
        off_sigma = parents_sigma
        if off_sigma < sigma_min:
            off_sigma = sigma_min

    return offspring, off_sigma


def mutation(xr, sigmar, limits, tau_prime, tau, sigma_min, mutation_strategy, mutation_prob=0.01):
    """Goal:
        The function performs mutation.
    -------------------------------------------------------
    Input:
        xr:
            Population, np.array
        sigmar:
            Sigma value(s), float/np.array
        limits:
            Lower and Upper Boundary of x_0, list
        tau_prime:
            tau_prime, float
        tau:
            tau, float
        sigma_min:
            Minimum sigma, float
        mutation_strategy:
            Mutation strategy, string
    -------------------------------------------------------
    Output:
        xm:
            Mutated population, np.array
        sigmar:
            Sigma value(s), float/np.array
    """

    # ---- Get mutation deviation
    if mutation_strategy == "non-uniform":
        mutation = np.random.normal(0, sigmar, size=xr.shape)
        # Mutate?
        mutation_chance = np.random.uniform(size=xr.shape)
        mutation[mutation_chance > mutation_prob] = 0
    elif mutation_strategy == "uniform":
        mutation = np.random.uniform(-sigmar, sigmar, size=xr.shape)
        # Mutate?
        mutation_chance = np.random.uniform(size=xr.shape)
        mutation[mutation_chance > mutation_prob] = 0
    elif mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
        # Normal distribution global
        nglob = tau_prime * np.random.normal(0, 1, size=xr.shape[0])
        # Initialize
        mutation = np.zeros(xr.shape)
        for i in range(xr.shape[0]):
            # Adapt sigma
            if mutation_strategy == "self-adaptive uncorrelated 1 stepsize":
                # Mutate sigma
                sigmam = sigmar[i] * np.exp(nglob[i])
                # Boundary
                if sigmam < sigma_min:
                    sigmam = sigma_min
                # Mutate?
                mutation_chance = np.random.uniform(0, 1)
                if mutation_chance > mutation_prob:
                    sigmar[i] = sigmar[i]
                else:
                    sigmar[i] = sigmam
                # Get mutation
                mutation_chance = np.random.uniform(size=xr.shape[1])
                mutation[i][mutation_chance <= mutation_prob] = sigmar[i] * np.random.normal(0, 1, size=xr.shape[1])
            else:
                # Mutate sigma
                sigmam = sigmar[i] * np.exp(nglob[i] + tau * np.random.normal(0, 1, size=xr.shape[1]))
                # Boundary
                sigmam[sigmam < sigma_min] = sigma_min
                # Mutate?
                mutation_chance = np.random.uniform(size=xr.shape[1])
                sigmar[i] = np.where(mutation_chance > mutation_prob, sigmar[i], sigmam)
                # Get mutation
                mutation[i][mutation_chance <= mutation_prob] = sigmar[i] * np.random.normal(0, 1, size=xr.shape[1])
    else:
        raise ValueError("Unknown mutation strategy.")

    # Get new population
    xm = xr + mutation

    # Adjust for boundaries
    xm[xm < limits[0]] = limits[0]
    xm[xm > limits[1]] = limits[1]
    xm = xm[:, 0, :]

    return xm, sigmar


def norm(x, pfit_pop):
    """Goal:
        Normalize fitness.
    ---------------------------------------------------------------------------------
    Input:
        x: Fitness value, float
        pfit_pop: Population fitness, np.array
    ---------------------------------------------------------------------------------
    Output:
        x_norm: Normalized fitness, float"""
    # If not all the same
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    # If negative
    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


def selection(pop, pop_sigma, fit_pop, npop, mutation_strategy, individual_gain, elitism=10, selection_strategy="Rank"):
    """Survival selection using (μ, λ) selection.
        Ref: Survivors selection p. 89 in Eiben and Smith (2015)
    ---------------------------------------------------------------------------------
    Input:
        pop: Population, np.array
        pop_sigma: Population sigma, np.array
        fit_pop: Population fitness, np.array
        npop: Number of parents, int
        mutation_strategy: Mutation strategy, string
    ---------------------------------------------------------------------------------"""
    if selection_strategy != "Rank":
        # Sort by fitness
        sorted_indices = np.argsort(fit_pop)
        # Reverse order
        sorted_indices = sorted_indices[::-1]
        # Sort
        pop, fit_pop, individual_gain = pop[sorted_indices, :], fit_pop[sorted_indices], individual_gain[sorted_indices]
        if type(pop_sigma) not in [float, int]:
            pop_sigma = pop_sigma[sorted_indices]

        # Avoiding negative probabilities, as fitness is ranges from negative numbers
        fit_pop_norm = np.array(list(map(lambda y: norm(y, fit_pop), fit_pop)))

        # Get probabilities
        probs = (fit_pop_norm[elitism:]) / (fit_pop_norm[elitism:]).sum()
        # Select indices
        if (npop - elitism) > 0:
            chosen = np.random.choice(np.arange(0, pop[elitism:, :].shape[0])
                                      , npop - elitism, p=probs, replace=False)
            chosen = chosen + elitism
        else:
            chosen = np.array([])
        # Add elitists
        if elitism > 0:
            best = np.arange(0, elitism)
            chosen = np.append(chosen, best)
            assert best[0] in chosen, "Elitism is not working correctly."

        # Sort chosen
        chosen = np.sort(chosen)

        # Get new population, sigma and its fitness
        pop, fit_pop, individual_gain = pop[chosen, :], fit_pop[chosen], individual_gain[chosen]
        if mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
            pop_sigma = pop_sigma[chosen]
    # elif selection_strategy == "Rank":
    #     if pop.shape[0] > npop:
    #         # Get offspring indices (last 2 * npop individuals in population)
    #         offspring_idx = np.arange(npop, pop.shape[0])

    #         # Select offspring
    #         offspring = pop[offspring_idx]

    #         # Get corresponding sigma and fitness values
    #         if mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
    #             sigma_offspring = pop_sigma[offspring_idx]
    #         else:
    #             sigma_offspring = pop_sigma
    #         fit_offspring = fit_pop[offspring_idx]
    #         fit_offspring_norm = fit_pop_norm[offspring_idx]

    #         # Select npop the best offspring idx
    #         if (npop - elitism) > 0:
    #             chosen = np.argsort(fit_offspring_norm)[::-1][0:(npop - elitism)]

    #         # Add elitists
    #         if elitism > 0:
    #             elitists_idx = np.argsort(fit_pop_norm)[::-1][0:elitism]
    #             # Select best offspring and fitness
    #             if (npop - elitism) > 0:
    #                 pop = np.append(offspring[chosen], pop[elitists_idx], axis=0)
    #                 fit_pop = np.append(fit_offspring[chosen], fit_pop[elitists_idx], axis=0)
    #                 if mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
    #                     pop_sigma = np.append(sigma_offspring[chosen], pop_sigma[elitists_idx], axis=0)
    #             else:
    #                 pop, fit_pop = pop[elitists_idx], fit_pop[elitists_idx]
    #                 if mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
    #                     pop_sigma = pop_sigma[elitists_idx]
    #         else:
    #             pop, fit_pop = offspring[chosen], fit_offspring[chosen]
    #             if mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
    #                 pop_sigma = sigma_offspring[chosen]

    return pop, pop_sigma, fit_pop, individual_gain


_n_evals = 0  # global variable to keep track of the number of evaluations


def simulation(env, x):
    """Goal:
        Simulate game and retrieve fitness. --> fitness function is already
        defined in environment.py
    ---------------------------------------------------------------------------------
    Input:
        env: Environment, class
        x: Weights, np array
    ---------------------------------------------------------------------------------
    Output:
        f: Fitness, float"""
    global _n_evals
    _n_evals += 1
    # fitness, player.life, enemy.life, time
    f, p, e, t = env.play(pcont=x)
    individual_gain = p - e
    return f, individual_gain


def evaluate(x):
    """Goal:
        Evaluate population and return fitness
    ---------------------------------------------------------------------------------
    Input:
        x:
            Population, np.array
    ---------------------------------------------------------------------------------
    Output:
        fitness:
            Fitness of population, np.array"""
    return np.array(list(map(lambda y: simulation(env, y), x)))


def search_best_solution(ngen, mu, n, limits, mag_sigma, n_offspring, mutation_strategy, tau_prime, tau, sigma_min,
                         smoothing, alfa, selection_strategy, elitism, mutation_prob,
                         parent_selection_strategy="Rank", printing=True):
    """Goal:
        Main run function.
    ---------------------------------------------------------------------------------
    Input:
        ..."""
    # ---- Initialize dict
    stats = {"fitness": [], "fitness_mean": [], "fitness_std": [], "individual_gain": [], "individual_gain_mean": [],
             "individual_gain_std": []}

    # ---- Initialize
    xm, sigma, = initialize(mu, n, limits, mag_sigma, mutation_strategy)

    # ---- Evaluate initial population
    evaluate_res = evaluate(xm)
    fitness = evaluate_res[:, 0]
    individual_gain = evaluate_res[:, 1]

    # ---- Loop through generations
    for gen in range(0, ngen):
        # Select parents
        if n_offspring > 0:
            parents, parents_sigma = parent_selection(xm, sigma, fitness, n_offspring, smoothing=smoothing,
                                                      parent_selection_strategy=parent_selection_strategy)

        # Crossover
        if n_offspring > 0:
            offspring, off_sigma = crossover(parents, parents_sigma, mutation_strategy, limits, alfa,
                                             sigma_min=sigma_min)
            # Mutate
            offspring, off_sigma = mutation(offspring, off_sigma, limits, tau_prime, tau, sigma_min,
                                            mutation_strategy=mutation_strategy, mutation_prob=mutation_prob)
            # Append offspring to population
            xm = np.concatenate((xm, offspring), axis=0)

            if mutation_strategy in ["self-adaptive uncorrelated 1 stepsize", "self-adaptive uncorrelated n stepsizes"]:
                sigma = np.concatenate((sigma, off_sigma), axis=0)
            # Evaluate new population
            evaluate_res = evaluate(xm)
            fitness = evaluate_res[:, 0]
            individual_gain = evaluate_res[:, 1]
        else:
            pass

        # Print generation n, the best fitness, mean fitness, std fitness
        if printing:
            print(f"--- Generation {gen}: best fitness {np.max(fitness):.2f}, mean fitness {np.mean(fitness):.2f}, "
                  f"std fitness {np.std(fitness):.2f}")

        # Select survivors
        xm, sigma, fitness, individual_gain = selection(pop=xm, pop_sigma=sigma, fit_pop=fitness, npop=mu,
                                                        mutation_strategy=mutation_strategy,
                                                        individual_gain=individual_gain, elitism=elitism,
                                                        selection_strategy=selection_strategy)

        # Save stats
        if mu > 1:
            stats["fitness"].append(max(fitness))
            stats["fitness_mean"].append(np.mean(fitness))
            stats["fitness_std"].append(np.std(fitness))
            best_from_pop_id = np.argmax(fitness)
            stats["individual_gain"].append(individual_gain[best_from_pop_id])
            stats["individual_gain_mean"].append(np.mean(individual_gain))
            stats["individual_gain_std"].append(np.std(individual_gain))
        else:
            stats["fitness"].append(fitness)
            stats["fitness_mean"].append(fitness)
            stats["fitness_std"].append(0)
            stats["individual_gain"].append(individual_gain)
            stats["individual_gain_mean"].append(individual_gain)
            stats["individual_gain_std"].append(0)

    # Return best solution and fitness data
    # best_solution = xm[np.argmax(fitness)]
    best_solution = xm[np.argmax(individual_gain)]

    # Assertion
    if elitism > 0:
        assert np.max(stats["fitness"]) == stats["fitness"][-1], "Elitism is not working correctly."

    return best_solution, stats


# Define
ngen = 30  # Number of generations
mu = 50  # Population size
n_offspring = mu  # Number of parent couples
limits = [-1, 1]  # Limits of x
mag_sigma = 0.2184  # Magnitude of sigma in case mutation strategy is uniform or non-uniform
sigma_min = 0.001  # Minimum value of sigma
smoothing = 1  # Smoothing parameter
alfa = 0.6335  # Crossover parameter
selection_strategy = "Not Rank"  # Rank
parent_selection_strategy = "Fitness Proportional"  # Rank or Fitness Proportional
elitism = 13  # Elitism, int --> if selection_strategy is not Rank
mutation_prob = 0.04  # Mutation probability

MUTATION_STRATEGY = "self-adaptive uncorrelated 1 stepsize"  # uniform, non-uniform, self-adaptive uncorrelated 1 stepsize, self-adaptive uncorrelated n stepsizes
# ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]  # all enemies
ENEMIES = [8]  # 3, 6, 7 are Selected enemies for the final report

REPEATS = 1  # Number of times to repeat the experiment

if __name__ == '__main__':
    print("Starting...")
    start_time = time.time()
    for enemy in ENEMIES:
        print(f"- Starting enemy #{enemy}")
        avg_fitness_max, avg_fitness_mean, avg_fitness_std = [], [], []
        avg_individual_gain_max, avg_individual_gain_mean, avg_individual_gain_std = [], [], []

        for repeat in range(1, REPEATS + 1):
            _n_evals = 0  # reset the number of evaluations
            print(f"-- Enemy: #{enemy}) Starting repeat #{repeat}")
            env = Environment(experiment_name=experiment_name,
                              enemies=[enemy],
                              playermode="ai",
                              multiplemode="no",
                              player_controller=player_controller(n_hidden_neurons),
                              enemymode="static",
                              level=2,
                              speed="fastest",
                              visuals=False)

            n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
            n = deepcopy(n_vars)

            # SET CONSTANTS
            if MUTATION_STRATEGY == "self-adaptive uncorrelated 1 stepsize":
                tau_prime, tau = 1 / np.sqrt(n), np.nan
            elif MUTATION_STRATEGY == "self-adaptive uncorrelated n stepsizes":
                tau_prime = 1 / np.sqrt(2 * n)
                tau = 1 / np.sqrt(2 * np.sqrt(n))
            else:
                tau_prime, tau = np.nan, np.nan

            # Run genetic algorithm
            best_solution, stats = search_best_solution(ngen, mu, n, limits, mag_sigma, n_offspring,
                                                        MUTATION_STRATEGY,
                                                        tau_prime, tau,
                                                        sigma_min, smoothing, alfa, selection_strategy,
                                                        elitism,
                                                        mutation_prob,
                                                        parent_selection_strategy=parent_selection_strategy
                                                        )
            fitness_max, fitness_mean, fitness_std = stats["fitness"], stats["fitness_mean"], stats["fitness_std"]
            individual_gain_max, individual_gain_mean, individual_gain_std = stats["individual_gain"], stats[
                "individual_gain_mean"], stats["individual_gain_std"]

            avg_fitness_max.append(fitness_max)
            avg_fitness_mean.append(fitness_mean)
            avg_fitness_std.append(fitness_std)
            avg_individual_gain_max.append(individual_gain_max)
            avg_individual_gain_mean.append(individual_gain_mean)
            avg_individual_gain_std.append(individual_gain_std)

            # Save the best solution
            # we save all the repeats as it may be interesting later to see if they differ and also we may want to
            # compose the whole initial population from the best solutions
            # save the stats dict to a file in {experiment_name}_enemy_{enemy}_{repeat}.pickle
            with open(f"{experiment_name}/stats_enemy_{enemy}_{repeat}.pickle", 'wb') as f:
                pickle.dump(stats, f)

            np.savetxt(f"{experiment_name}/best_enemy_{enemy}_{repeat}.txt", best_solution)
            verify_solution(env, best_solution)

        # make vector of average fitness values
        fitness_max = np.average(np.array(avg_fitness_max), axis=0)
        fitness_mean = np.average(np.array(avg_fitness_mean), axis=0)
        fitness_std = np.average(np.array(avg_fitness_std), axis=0)
        individual_gain_max = np.average(np.array(avg_individual_gain_max), axis=0)
        individual_gain_mean = np.average(np.array(avg_individual_gain_mean), axis=0)
        individual_gain_std = np.average(np.array(avg_individual_gain_std), axis=0)

        # Plot results
        plt.plot(fitness_max, label="Max Fitness Value")
        plt.plot(fitness_mean, label="Mean Fitness Value")
        plt.plot(individual_gain_max, label="Max Individual Gain")
        plt.plot(individual_gain_mean, label="Mean Individual Gain")

        # plt standard deviation around mean fitness value
        plt.fill_between(np.arange(0, len(fitness_mean)),
                         np.array(fitness_mean) - np.array(fitness_std),
                         np.array(fitness_mean) + np.array(fitness_std), alpha=0.5)
        plt.fill_between(np.arange(0, len(individual_gain_mean)),
                         np.array(individual_gain_mean) - np.array(individual_gain_std),
                         np.array(individual_gain_mean) + np.array(individual_gain_std), alpha=0.5)

        # plot box plot with max fitness value, mean fitness value, std fitness value
        plt.legend()
        plt.title(f"Specialist solution for enemy #{enemy}")
        plt.xlabel("Generation")
        plt.ylabel(f"Fitness")
        # plt.ylim(0, 100)
        # plt.text(0.5, -0.05, f'Individuals evaluated: {_n_evals}', ha='center')
        plt.show()

    print("\n\nDone!")
    print(f"--- Took {time.time() - start_time} seconds ---")

# OPTUNA OPTIMIZATION
# def objective(trial, mutation_strategy, n, enemies):
#     # Define parameters
#     #budget = 30 * 50
#     ngen = 30 #trial.suggest_categorical("ngen", [10, 20, 30, 50, 100])
#     mu = 50 #int(budget / ngen)
#     #assert (mu * ngen) == budget, "Budget is not correct."

#     n_offspring = mu
#     mag_sigma = trial.suggest_float("mag_sigma", 0, 1)
#     sigma_min = 0 #trial.suggest_float("sigma_min", 0, 1) --> mutation specific
#     alfa = trial.suggest_float("alfa", 0, 1)
#     selection_strategy = "Not Rank"
#     parent_selection_strategy = "Fitness Proportional"
#     elitism = trial.suggest_int("elitism", 0, mu)
#     mutation_prob = trial.suggest_float("mutation_prob", 0, 1)
#     tau = np.nan # --> mutation specific
#     tau_prime = np.nan # --> mutation specific
#     # --- Perform Run(s)
#     scores = []
#     for enemy in enemies:
#         # Set environment
#         env.update_parameter('enemies', [enemy])
#         for run in range(3):
#             print("Subrun ", run)
#             # Search best solution
#             best_solution, stats = search_best_solution(ngen, mu, n, limits, mag_sigma, n_offspring,
#                                                                 mutation_strategy,
#                                                                 tau_prime, tau,
#                                                                 sigma_min, smoothing, alfa, selection_strategy,
#                                                                 elitism,
#                                                                 mutation_prob,
#                                                                 parent_selection_strategy=parent_selection_strategy, printing = False
#                                                                 )
#             # Add score
#             scores.append(stats["fitness"][-1])


#     return np.median(scores)
#
#
# # ---- Initialize environment
# env = Environment(experiment_name=experiment_name,
#                     enemies=[3],
#                     playermode="ai",
#                     player_controller=player_controller(n_hidden_neurons),
#                     enemymode="static",
#                     level=2,
#                     speed="fastest",
#                     visuals=False)
# # Set parameters
# study_name = "Optimize"
# nruns = 100
# mutation_strategy = "non-uniform"
# n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
# n = deepcopy(n_vars)
# enemies = [3, 6, 7]
#
# # ---- Create study
# study = optuna.create_study(direction = "maximize", study_name = study_name)
# def obj(trial):
#     return objective(trial, mutation_strategy, n, enemies)
#
# # ---- Optimize
# study.optimize(obj, n_trials = nruns, show_progress_bar=True)
#
# # ---- Save Study
# date = dt.datetime.now().strftime("%m-%d_%H_%M")
# print(f"Optimal value: {study.best_value: .4f}")
# with open(f"optuna_run_{date}.pickle", 'wb') as f:
#     pickle.dump(study, f)
#
# print("Done!")

# # -------------------------------------------------------------------------------------------
# # Set path
# path = "final_specialist_assignment\\1 step size\\"
# # Initialize results
# results = {}

# # Get Results
# for i in range(0, 10):
#     for enemy in [3, 6, 7]:
#         # Initialize
#         results[f"enemy_{enemy}_{i + 1}"] = {"fitness": [], "gain": []}
#         # Open text file
#         x = np.loadtxt(path + f"best_enemy_{enemy}_{i + 1}.txt")
#         # Set environment
#         env = Environment(experiment_name=experiment_name,
#                               enemies=[enemy],
#                               playermode="ai",
#                               player_controller=player_controller(n_hidden_neurons),
#                               enemymode="static",
#                               level=2,
#                               speed="fastest",
#                               visuals=False)
#         # Run Simulation
#         for j in range(0, 5):
#             f, gain = simulation(env, x)
#             # Save results
#             results[f"enemy_{enemy}_{i + 1}"]["fitness"].append(f)
#             results[f"enemy_{enemy}_{i + 1}"]["gain"].append(gain)


# with open("results.pickle", 'wb') as f:
#     pickle.dump(results, f)
