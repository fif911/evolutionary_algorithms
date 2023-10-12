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

import numpy as np
from evoman.environment import Environment
from scipy.spatial.distance import pdist

import pymoo.gradient.toolbox as anp
from nn_crossover import NNCrossover
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.visualization.scatter import Scatter
from utils import simulation, verify_solution, init_env

# np.random.seed(1)

global ENEMIES
global CLUSTER
global ALL_ENEMIES
global N_GENERATIONS
global POP_SIZE

n_hidden_neurons = 10

experiment_name = 'pymoo_sms_emoa'
solution_file_name = 'pymoo_sms_emoa_best.txt'
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
                                                          for ind_id in range(POP_SIZE)]

        for ienemy, enemy in enumerate(ENEMIES):
            objectives_fitness[f"objective_{ienemy + icl + 2}"] = dict_enemies[enemy]

        out["F"] = anp.column_stack([objectives_fitness[key] for key in objectives_fitness.keys()])


def main(env: Environment, n_genes: int, population=None):
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=[1, 2, 3, 4, 5, 6, 7, 8],
        n_objectives=len(ENEMIES) + (len(CLUSTER))
    )
    mutation_prob = 0.1
    mutation_sigma = 1
    crossover_prob = 0.6

    if population is None:
        algorithm = SMSEMOA(pop_size=POP_SIZE, crossover=NNCrossover(prob=crossover_prob),
                            mutation=GaussianMutation(prob=mutation_prob, sigma=mutation_sigma))  # , seed=1
    else:
        population = np.array(population)
        algorithm = SMSEMOA(pop_size=POP_SIZE, sampling=population, crossover=NNCrossover(prob=crossover_prob),
                            mutation=GaussianMutation(prob=mutation_prob, sigma=mutation_sigma))  # , seed=1
    # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
    algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), verbose=False)

    step = 1
    while algorithm.has_next():
        print(np.round((step / N_GENERATIONS * 100), 0), "%", end="\r")
        pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop)
        algorithm.tell(infills=pop)
        step += 1

    # obtain the result objective from the algorithm
    res = algorithm.result()

    res.F = 1 / res.F

    max_enemies_beaten = 0
    best_solutions = []
    best_not_beaten = []
    best_x = []
    env.update_parameter('level', 2)
    for i, x in enumerate(res.X):
        enemies_beaten, enemies_not_beaten, _ = verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8], verbose=True,
                                                                print_results=False)
        if len(enemies_beaten) > max_enemies_beaten:
            max_enemies_beaten = len(enemies_beaten)
            best_solutions = [x]  # reset the list because we found a better performing solution
            best_not_beaten = [enemies_not_beaten]
            best_x = [x]
        elif len(enemies_beaten) == max_enemies_beaten:
            best_solutions.append(x)  # add to the list the solution that beats the same number of enemies
            best_not_beaten.append(enemies_not_beaten)
            best_x.append(x)

    # # save the best solutions to files
    # for i, solution in enumerate(best_solutions):
    #     np.savetxt(f'{experiment_name}/{solution_file_name}_{i}', solution)

    return [i.x for i in algorithm.ask()], best_not_beaten, best_x


if __name__ == '__main__':
    time_start = time.time()

    # --------------------------- The BEGINNING (3, 5, 7, 8)
    # N_GENERATIONS = 30
    # POP_SIZE = 90

    # CLUSTER = [[5]]
    # ENEMIES = np.array([3, 7])
    # ALL_ENEMIES = CLUSTER[0] + list(ENEMIES)

    N_GENERATIONS = 1
    POP_SIZE = 100

    CLUSTER = [[1]]
    ENEMIES = np.array([2, 3, 4, 5, 6, 7, 8])
    ALL_ENEMIES = CLUSTER[0] + list(ENEMIES)

    env, n_genes = init_env(experiment_name, ENEMIES, n_hidden_neurons)
    env.update_parameter('multiplemode', 'no')
    env.update_parameter('level', 2)

    pop, best_not_beaten, best_x = main(env, n_genes)
    EVALUATIONS = N_GENERATIONS * POP_SIZE

    # Save population
    POP = copy.deepcopy(pop)

    print("Training Round 6", end="\r")
    N_GENERATIONS = 10
    POP_SIZE = 20

    nhistory = 10  # For beaten2
    beaten = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
              6: 0, 7: 0, 8: 0}
    beaten2 = {1: nhistory * [0], 2: nhistory * [0], 3: nhistory * [0], 4: nhistory * [0], 5: nhistory * [0],
               6: nhistory * [0], 7: nhistory * [0], 8: nhistory * [0]}
    best_performing = 0
    BEST_x = ""

    iterations = 0
    while ENEMIES.size != 0:
        # Select population
        idx_pop = np.random.choice(range(len(POP)), size=POP_SIZE, replace=False)
        pop = [POP[idx] for idx in idx_pop]
        POP = [POP[idx] for idx in range(len(POP)) if idx not in idx_pop]

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
        # CLUSTER = [[enemy for enemy in range(1, 9) if enemy not in best_not_beaten[0]]]

        try:
            ALL_ENEMIES = [enemy for cl in CLUSTER for enemy in cl] + list(ENEMIES)
        except TypeError:
            ALL_ENEMIES = [enemy for enemy in CLUSTER] + [ENEMIES]
        print(f"Cluster: {CLUSTER}")
        print(f"Enemies: {ENEMIES}")
        pop, best_not_beaten, best_x = main(env, n_genes, population=pop)

        # Update number of evaluations
        destroyed = [enemy for enemy in np.arange(1, 9) if enemy not in best_not_beaten[0]]
        if len(destroyed) > best_performing:
            best_performing = len(destroyed)
            BEST_x = best_x[0]
            np.savetxt("BEST_SOLUTION.txt", BEST_x)
        for enemy in range(1, 9):
            if enemy in destroyed:
                beaten[enemy] += 1
                beaten2[enemy] = beaten2[enemy][1:] + [1]
            else:
                beaten2[enemy] = beaten2[enemy][1:] + [0]

        print(f"\tEnemies beaten: {8 - len(best_not_beaten[0])}")
        print("\tBeaten:\n\t\t", beaten)
        print("\t STD of Beaten last iterations:\n\t\t", [np.std(beaten2[enemy]).round(2) for enemy in np.arange(1, 9)])
        print("\tDiversity: ", np.mean(pdist(pop, metric="euclidean")))
        print("\tCurrent Record: ", best_performing)
        print("----")

        # Save population
        POP += copy.deepcopy(pop)
        print("\tPopulation Diversity: ", np.mean(pdist(POP, metric="euclidean")))
        EVALUATIONS += N_GENERATIONS * POP_SIZE
        print("\tEvaluations: ", EVALUATIONS)
        iterations += 1

    print(f"Total time (minutes): {(time.time() - time_start) / 60:.2f}")
    print("Done!")
