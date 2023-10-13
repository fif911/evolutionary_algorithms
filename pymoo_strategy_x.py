"""
Implementation of multi-objective optimisation using pymoo

https://pymoo.org/algorithms/list.html


Algorithm: SMS-EMOA
Algorithm paper: https://sci-hub.se/https://doi.org/10.1016/j.ejor.2006.08.008
Docs link: https://pymoo.org/algorithms/moo/sms.html
"""
import copy
import os
import time
import uuid

import numpy as np
from evoman.environment import Environment
from numpy.lib.arraysetops import unique
from scipy.spatial.distance import pdist

import pymoo.gradient.toolbox as anp
from nn_crossover import NNCrossover
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.operators.mutation.gauss import GaussianMutation
from utils import simulation, verify_solution, init_env, read_solutions_from_file, fitness_proportional_selection, \
    tournament_selection

# np.random.seed(1)

global ENEMIES
global CLUSTER
global ALL_ENEMIES
global N_GENERATIONS
global POP_SIZE
global SEEN_8_BEATING_SOLUTIONS
n_hidden_neurons = 10

experiment_name = 'magic_8'
solution_file_name = 'strategy_x.txt'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

os.environ["SDL_VIDEODRIVER"] = "dummy"


class objectives(Problem):
    enemies: list[int]
    env: Environment

    def __init__(self, env: Environment, n_genes: int, enemies: list[int], n_objectives):
        self.env = env
        self.enemies = enemies
        super().__init__(n_var=n_genes, n_obj=n_objectives, xl=-1, xu=1, type_var=float)

    def _evaluate(self, x: list[np.array], out, *args, **kwargs):
        """Evaluate the fitness of each individual in the population
        We can turn on elementwise_evaluation to evaluate each individual in the population separately
        https://pymoo.org/problems/parallelization.html#Custom-Parallelization

        x - list of individuals in the population
        out - dictionary with the fitness outputs

        # when we have multiple enemies we need to average the fitness somehow
        # Ideas: mean of the fitness will show how agent performs with both enemies
        #        max will show how agent performs with the worst enemy (this does not reflect the performance with
        #        another enemy)
        #        weighted average is another option, but then we have another problem of how to weight the enemies
        """

        # Initialize
        dict_enemies = {}
        # Get fitness for each enemy
        for enemy in ALL_ENEMIES:
            self.env.update_parameter('enemies', [enemy])

            dict_enemies[enemy] = []
            for individual_id in range(len(x)):
                dict_enemies[enemy].append(simulation(self.env, x[individual_id], inverted_fitness=True))

        objectives_fitness = {}
        for icl, cl in enumerate(CLUSTER):
            objectives_fitness[f"objective_{icl + 1}"] = [np.max([dict_enemies[enemy_id][ind_id] for enemy_id in cl])
                                                          for ind_id in range(len(x))]

        for ienemy, enemy in enumerate(ENEMIES):
            objectives_fitness[f"objective_{ienemy + icl + 2}"] = dict_enemies[enemy]

        out["F"] = anp.column_stack([objectives_fitness[key] for key in objectives_fitness.keys()])


def main(env: Environment, n_genes: int, population=None,
         mutation_prob=1.0,
         mutation_sigma=1.0,
         crossover_prob=1.0,
         crossover_class=NNCrossover
         ):
    # env.update_parameter('randomini', 'yes')
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=[1, 2, 3, 4, 5, 6, 7, 8],
        n_objectives=len(ENEMIES) + (len(CLUSTER))
    )

    if population is None:
        algorithm = SMSEMOA(pop_size=POP_SIZE, crossover=crossover_class(prob=crossover_prob),
                            mutation=GaussianMutation(prob=mutation_prob, sigma=mutation_sigma))  # , seed=1
    else:
        population = np.array(population)
        algorithm = SMSEMOA(pop_size=POP_SIZE, sampling=population, crossover=crossover_class(prob=crossover_prob),
                            mutation=GaussianMutation(prob=mutation_prob, sigma=mutation_sigma))  # , seed=1
    # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
    algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), verbose=False)

    step = 1
    while algorithm.has_next():
        print(np.round((step / N_GENERATIONS * 100), 0), "%", end="\r")
        pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop, skip_already_evaluated=False)
        algorithm.tell(infills=pop)
        step += 1

    # obtain the result objective from the algorithm
    res = algorithm.result()

    max_enemies_beaten = 0
    best_solutions = []
    best_solutions_do_not_beat = []
    env.update_parameter('level', 2)
    for i, x in enumerate(res.X):
        enemies_beaten, enemies_not_beaten, _ = verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8], verbose=True,
                                                                print_results=False)
        if len(enemies_beaten) > max_enemies_beaten:
            max_enemies_beaten = len(enemies_beaten)
            best_solutions = [x]  # reset the list because we found a better performing solution
            best_solutions_do_not_beat = [enemies_not_beaten]
        elif len(enemies_beaten) == max_enemies_beaten:
            best_solutions.append(x)  # add to the list the solution that beats the same number of enemies
            best_solutions_do_not_beat.append(enemies_not_beaten)

    # # save the best solutions to files
    if max_enemies_beaten == 8:
        for i, solution in enumerate(best_solutions):
            if list(solution) not in SEEN_8_BEATING_SOLUTIONS:
                SEEN_8_BEATING_SOLUTIONS.append(list(solution))
                np.savetxt(f'{experiment_name}/beats_8_{uuid.uuid4()}_{solution_file_name}', solution)
        print("Unique solutions beating 8 enemies: ", len(SEEN_8_BEATING_SOLUTIONS))

    # combine the best solutions with the rest of the population and select the best ones
    population_with_pareto_front_solutions, idx = \
        np.unique(np.concatenate((algorithm.result().X, algorithm.pop.get("X"))), axis=0, return_index=True)
    print("UNIQUE Population with pareto front solutions: ", len(population_with_pareto_front_solutions))
    print(algorithm.result().X == algorithm.pop.get("X"))
    population_with_pareto_front_fitness = np.concatenate((algorithm.result().F, algorithm.pop.get("F")))[idx]
    population_with_pareto_front_fitness = np.mean(population_with_pareto_front_fitness, axis=1)
    # resulting_population = unique(resulting_population, axis=0)[:POP_SIZE]
    survived_pop = tournament_selection(population_with_pareto_front_solutions,
                                        population_with_pareto_front_fitness,
                                        POP_SIZE, k=5)

    return list(survived_pop), best_solutions_do_not_beat, best_solutions


if __name__ == '__main__':
    SMART_INIT = False
    SEEN_8_BEATING_SOLUTIONS = []
    time_start = time.time()

    N_GENERATIONS = 1
    POP_SIZE = 100

    CLUSTER = [[1]]
    ENEMIES = np.array([2, 3, 4, 5, 6, 7, 8])
    ALL_ENEMIES = CLUSTER[0] + list(ENEMIES)

    env, n_genes = init_env(experiment_name, ENEMIES, n_hidden_neurons)
    env.update_parameter('multiplemode', 'no')
    env.update_parameter('level', 2)

    pop, best_not_beaten, best_x = [], [[]], []
    if SMART_INIT:
        EVALUATIONS = N_GENERATIONS * POP_SIZE
        # load population from solutions beats 5 enemies
        solutions_ = read_solutions_from_file("./farmed_beats_8")
        # pop = np.concatenate((unique(solutions_, axis=0), pop))
        pop = list(solutions_[:POP_SIZE])
    else:
        pop = np.random.uniform(-1, 1, (POP_SIZE, n_genes))
        EVALUATIONS = 0
    # Save population
    POP = copy.deepcopy(pop)

    N_GENERATIONS = 5
    POP_SIZE = 20

    nhistory = 10  # For beaten2
    beaten = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
              6: 0, 7: 0, 8: 0}
    beaten2 = {1: nhistory * [0], 2: nhistory * [0], 3: nhistory * [0], 4: nhistory * [0], 5: nhistory * [0],
               6: nhistory * [0], 7: nhistory * [0], 8: nhistory * [0]}
    best_performing = 0

    iterations = 0
    while ENEMIES.size != 0:
        print(" ---- Iteration ", iterations, " ----")
        print("Population size: ", len(POP), " POP_SIZE: ", POP_SIZE)
        # Randomly choose 20 individuals from the whole population and train them
        idx_pop = np.random.choice(range(len(POP)), size=POP_SIZE, replace=False)
        pop = [POP[idx] for idx in idx_pop]  # population  to train
        POP = [POP[idx] for idx in range(len(POP)) if idx not in idx_pop]  # rest of the population

        # Set enemies
        beaten_vals = np.array([beaten[enemy] for enemy in np.arange(1, 9)])
        if sum(beaten_vals) == 0:
            probs = np.ones(8) / 8
        else:
            probs = sum(beaten_vals) / np.where(beaten_vals == 0, 0.01, beaten_vals)
            probs = probs / sum(probs)

        opponents = np.random.choice(np.arange(1, 9), p=probs, size=3, replace=False)
        CLUSTER = [[opponents[0]]]
        ENEMIES = np.array([opponents[1], opponents[2]])

        try:
            ALL_ENEMIES = [enemy for cl in CLUSTER for enemy in cl] + list(ENEMIES)
        except TypeError:
            ALL_ENEMIES = [enemy for enemy in CLUSTER] + [ENEMIES]
        print(f"Cluster: {CLUSTER}")
        print(f"Enemies: {ENEMIES}")
        # the better we perform -> less mutation and crossover rate we need
        # mutation_prob ranges from 0.1 to 1
        mutation_prob = 1 - ((8 - len(best_not_beaten[0])) / 8) * 0.95
        mutation_sigma = 1 - ((8 - len(best_not_beaten[0])) / 8) * 0.95
        crossover_prob = 1 - ((8 - len(best_not_beaten[0])) / 8) * 0.95
        print(f"Mutation prob: {mutation_prob}")
        print(f"Mutation sigma: {mutation_sigma}")
        print(f"Crossover prob: {crossover_prob}")

        pop, best_not_beaten, best_x = main(env, n_genes, population=pop, mutation_prob=mutation_prob,
                                            mutation_sigma=mutation_sigma, crossover_prob=crossover_prob)

        # Update number of evaluations
        # list of indexes of enemies that were beaten by FIRST best performing ind
        destroyed = [enemy for enemy in np.arange(1, 9) if enemy not in best_not_beaten[0]]
        if len(destroyed) > best_performing:
            best_performing = len(destroyed)
            BEST_x = best_x[0]
            np.savetxt("BEST_SOLUTION.txt", BEST_x)
        for enemy in range(1, 9):
            if enemy in destroyed:
                beaten[enemy] += 1
                beaten2[enemy] = beaten2[enemy][1:] + [1]  # remove oldest value and add 1
            else:
                beaten2[enemy] = beaten2[enemy][1:] + [0]  # remove oldest value and add 0

        print(f"\tEnemies beaten: {8 - len(best_not_beaten[0])}")
        print("\tBeaten:\n\t\t", beaten)
        print("\t STD of Beaten last iterations:\n\t\t", [np.std(beaten2[enemy]).round(2) for enemy in np.arange(1, 9)])
        print("\tDiversity: ", np.mean(pdist(pop, metric="euclidean")))
        print("\tCurrent Record: ", best_performing)

        # Save population
        POP += copy.deepcopy(pop)
        print("\tPopulation Diversity: ", np.mean(pdist(POP, metric="euclidean")))
        EVALUATIONS += N_GENERATIONS * POP_SIZE
        print("\tEvaluations: ", EVALUATIONS)
        iterations += 1

    print(f"Total time (minutes): {(time.time() - time_start) / 60:.2f}")
    print("Done!")
