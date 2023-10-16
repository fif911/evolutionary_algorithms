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
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GaussianMutation
# from pymoo.visualization.scatter import Scatter
from utils import simulation, verify_solution, init_env

# Settings
N_REPEATS = 10
MAX_EVALUATIONS = 50_000
N_GENERATIONS = 10
POP_SIZE = 20  # Subpopulation
WHOLE_POP_SIZE = 100  # whole population

pmut, vsigma, pcross = 1, 1, 1  # Mutation probability, mutation strength, crossover probability
crossovermode = "NN"  # NN or SBX

min_n_enemies = 3  # Minimum number of enemies to train on
max_n_enemies = 3  # Maximum number of enemies to train on

nhistory = 10  # For keeping track of standard deviation of beaten enemies
n_hidden_neurons = 10

# Global variables
global ENEMIES
global CLUSTER

# Create experiment folder
experiment_name = 'pymoo_sms_emoa'
solution_file_name = 'pymoo_sms_emoa_best.txt'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Set SDL to use the dummy NULL video driver, so it doesn't need a windowing system.
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
        """

        # Initialize
        dict_enemies = {}
        # Get fitness for each enemy
        for enemy in self.enemies:
            # Update the enemy
            self.env.update_parameter('enemies', [enemy])
            # Initialize the dictionary for this enemy
            dict_enemies[enemy] = []
            # Get the fitness for each individual
            for individual_id in range(len(x)):
                if self.env.randomini == "no":
                    dict_enemies[enemy].append(simulation(self.env, x[individual_id], inverted_fitness=True))
                else:
                    sims = []
                    for rep_rand in range(0, 5):  # Repeat multiple times to get a more accurate fitness
                        sims.append(simulation(self.env, x[individual_id], inverted_fitness=True))
                    dict_enemies[enemy].append(np.mean(sims))
        # If we have clusters, get the fitness for each cluster
        objectives_fitness = {}
        for icl, cl in enumerate(CLUSTER):
            objectives_fitness[f"objective_{icl + 1}"] = [np.max([dict_enemies[enemy_id][ind_id] for enemy_id in cl])
                                                          for ind_id in range(len(x))]

        # Get the fitness for each enemy
        for ienemy, enemy in enumerate(ENEMIES):
            objectives_fitness[f"objective_{ienemy + 2}"] = dict_enemies[enemy]

        # Get the fitness for the whole population and all objectives
        out["F"] = anp.column_stack([objectives_fitness[key] for key in objectives_fitness.keys()])


def main(env: Environment, n_genes: int, population=None, pmut=1, vsigma=1, pcross=1, crossovermode="NN",
         algorithm=None):
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=ALL_ENEMIES,
        n_objectives=len(ENEMIES) + (len(CLUSTER) > 0)
    )

    # --- Set crossover mode
    if crossovermode == "NN":
        crossover = NNCrossover(prob=pcross)
    elif crossovermode == "SBX":
        crossover = SBX(prob=pcross)
    else:
        raise ValueError("Crossover mode not recognized")

    # --- Set the algorithm if it is not passed
    if algorithm is None:
        if population is None:  # No population passed
            algorithm = SMSEMOA(pop_size=POP_SIZE, crossover=crossover,
                                mutation=GaussianMutation(prob=pmut, sigma=vsigma))  # , seed=1
        else:
            # Set population
            population = np.array(population)
            # Create algorithm
            algorithm = SMSEMOA(pop_size=POP_SIZE, sampling=population, crossover=crossover,
                                mutation=GaussianMutation(prob=pmut, sigma=vsigma))  # , seed=1
            algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), verbose=False)
        algo_label = False
    else:
        algo_label = True

    # --- Run te algorithm
    step = 0
    while algorithm.has_next():
        print("\t\t", np.round((step / N_GENERATIONS * 100), 0), "%", end="\r")
        if (step == 0) and (algo_label == True):  # If we are in the first step + we have an algorithm
            pop = Population(population)
        else:
            pop = algorithm.ask()

        algorithm.evaluator.eval(problem, pop)
        algorithm.tell(infills=pop)
        # Increase step
        step += 1

    # --- Check results
    # Initialize
    best_x, max_enemies_beaten, best_enemies = [], 0, np.zeros(8)

    # Obtain the result objective from the algorithm
    res = algorithm.result()

    # Obtain the best solutions in terms of enemies beaten
    for i, x in enumerate(res.X):
        # Get enemy lives
        enemies_beaten, enemies_not_beaten, enemy_lives = verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                                                                          verbose=True,
                                                                          print_results=False)
        enemy_lives = np.array(enemy_lives)
        # Get the number of enemies beaten
        if len(enemies_beaten) > max_enemies_beaten:  # New record
            max_enemies_beaten = len(enemies_beaten)
            best_x = [x]
            best_enemies = np.where(enemy_lives == 0, 1, 0)
        elif len(enemies_beaten) == max_enemies_beaten:  # Equal record
            best_x.append(x)
            best_enemies += np.where(enemy_lives == 0, 1, 0)

    # Normalize
    best_enemies = best_enemies / len(best_x)

    return [i.X for i in algorithm.ask()], best_x, max_enemies_beaten, best_enemies, algorithm


if __name__ == '__main__':
    for trial in range(0, N_REPEATS):
        time_start = time.time()
        # Initialize
        EVALUATIONS, ENEMIES = 0, np.array([1])
        # Environment
        env, n_genes = init_env(experiment_name, [1], n_hidden_neurons)
        env.update_parameter('multiplemode', 'no')
        env.update_parameter('randomini', 'no')
        env.update_parameter("level", 2)
        print("----------------------------------------------------------------------------------")
        # --------------------------- Initialize population
        # Sample random initial population
        pop = np.random.uniform(-1, 1, size=(WHOLE_POP_SIZE, 265))
        # Save population and population size
        POP, popsize_or = copy.deepcopy(pop), copy.deepcopy(len(pop))

        # --------------------------- Iterated Learning/Constrained Led Approach
        assert nhistory % 2 == 0, "nhistory must be even"
        # Initialize
        iterations = 0  # Number of iterations
        best_performing = 0  # Most enemies beaten by a single individual
        best_performing_array = []  # Most enemies beaten by a single individual over time
        BEST_x = ""  # Best individual

        FITNESS = np.zeros((int(8 * WHOLE_POP_SIZE), int(MAX_EVALUATIONS / (POP_SIZE * N_GENERATIONS) + 1)))  # fITNESS

        algos = {}  # Cache algorithms
        # Number of enemies beaten in current population
        beaten = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        # Number of enemies beaten in current population over time nhistory
        beaten2 = {1: nhistory * [0], 2: nhistory * [0], 3: nhistory * [0], 4: nhistory * [0], 5: nhistory * [0],
                   6: nhistory * [0], 7: nhistory * [0],
                   8: nhistory * [0]}
        # Average number of enemies beaten in current Pareto front for the most generalizable enemies
        beaten3 = np.zeros(8)

        while EVALUATIONS < MAX_EVALUATIONS:
            # ---- Evaluate population --> beat rates
            # Set to zero/remove oldest value
            for enemy in range(1, 9):
                beaten[enemy] = 0
                beaten2[enemy] = beaten2[enemy][1:] + [0]

            # Evaluate population
            most_beaten, most_x = 0, ""
            for i_x, x in enumerate(POP):
                # Set to zero for each member
                enemy_beaten = 0
                for enemy in range(1, 9):
                    # Simulate
                    env.update_parameter('enemies', [enemy])
                    p, e, t = simulation(env, x, verbose=True)
                    # Get fitness
                    f = (100 - e) + np.log(p + 0.001)
                    # Store fitness
                    FITNESS[int(i_x + (enemy - 1) * WHOLE_POP_SIZE), iterations] = f
                    # Update beaten
                    if (e == 0) and (p > 0):
                        beaten[enemy] += 1
                        beaten2[enemy][-1] += 1
                        enemy_beaten += 1
                    if i_x == (len(POP) - 1):  # Last one --> Normalize
                        beaten2[enemy][-1] = beaten2[enemy][-1] / popsize_or * 100
                        beaten[enemy] = beaten[enemy] / popsize_or * 100
                # Reset
                if enemy_beaten >= most_beaten:
                    most_beaten = enemy_beaten
                    best_x = copy.deepcopy(x)

            # Update best performing
            if most_beaten >= best_performing:
                best_performing = copy.deepcopy(most_beaten)
            # Append to best_performing
            best_performing_array.append(most_beaten)

            # --- Print some settings
            print("NEW ITERATION: ", iterations)
            print("Population Size: ", POP_SIZE)
            print("Mutation Probability: ", pmut)
            print("Crossover Type: ", crossovermode)
            print("\tCurrent Record: ", best_performing)
            print("\tBeaten Percentages Current Population [%] and STD over " + str(nhistory) + " Runs [%]")
            for enemy in range(1, 9):
                print("\t\tEnemy: ", enemy)
                print("\t\t\t" + str(np.round(beaten[enemy], 2)) + "% - " + str(np.std(beaten2[enemy]).round(2)))
            print("\tPopulation Diversity: ", np.mean(pdist(POP, metric="euclidean")))

            # Select random part of the population
            idx_pop = np.random.choice(range(len(POP)), size=POP_SIZE, replace=False)
            pop = [POP[idx] for idx in idx_pop]
            POP = [POP[idx] for idx in range(len(POP)) if idx not in idx_pop]

            print("\tOld Diversity of Subsample: ", np.mean(pdist(pop, metric="euclidean")))

            # Set enemies acccording to probabilities
            beaten_vals = beaten3
            if sum(beaten_vals) == 0:
                probs = np.ones(8) / 8
            else:
                min_val = np.min(np.where(beaten_vals == 0, 999, beaten_vals))
                probs = sum(beaten_vals) / np.where(beaten_vals == 0, min_val, beaten_vals)
                probs = probs / sum(probs)

            # Choose 3 enemies with inverse probabilities.
            # Enemies on which the population performs poorly will be chosen with higher probability
            n_enemies = np.random.choice([min_n_enemies, max_n_enemies])
            opponents = np.random.choice(np.arange(1, 9), p=probs, size=n_enemies, replace=False)
            CLUSTER = [[opponents[0]]]
            ENEMIES = np.array([opponents[i] for i in range(1, n_enemies)])

            try:
                ALL_ENEMIES = [enemy for cl in CLUSTER for enemy in cl] + list(ENEMIES)
            except TypeError:
                ALL_ENEMIES = [enemy for enemy in CLUSTER] + [ENEMIES]

            print("\tTraining on: ")
            print(f"\t\tEnemy 1: {CLUSTER}")
            print(f"\t\tEnemy 2,3: {ENEMIES}")

            # Create unique identifier for the algorithm instance
            ALL_ENEMIES.sort()  # sort for hashing reproducibility
            algorithm_hash = "-".join([str(i) for i in ALL_ENEMIES])

            # Check if algorithm was already initialised and run
            if algorithm_hash in algos.keys():
                pop, best_x, max_enemies_beaten, best_enemies, algorithm = main(env, n_genes, population=pop, pmut=pmut,
                                                                                vsigma=vsigma, pcross=pcross,
                                                                                crossovermode=crossovermode,
                                                                                algorithm=algos[algorithm_hash])
            else:
                pop, best_x, max_enemies_beaten, best_enemies, algorithm = main(env, n_genes, population=pop, pmut=pmut,
                                                                                vsigma=vsigma, pcross=pcross,
                                                                                crossovermode=crossovermode)
            # Cache algorithm instance --> don't tab this one, should be saved in both instances!!!!!!!!!!!!!!!!!!!!!!!
            algos = {algorithm_hash: algorithm}

            # Increase beaten3
            beaten3 += best_enemies
            print(beaten3)

            # Check for best performing in Pareto front --> before we saved these x-values, but now we don't because of speed considerations
            # We use an archived based approach, so this is allowed. However, it also means that these solution might not end up in our plots
            # , so the might differ because we have some spacing (Ngeneration * Popsize) between storage of fitness. But we cannot store all fitness values
            if max_enemies_beaten > best_performing:
                best_performing = max_enemies_beaten
            elif max_enemies_beaten == best_performing:
                pass
            # Append to best_performing
            if max_enemies_beaten > most_beaten:
                best_performing_array[-1] = copy.deepcopy(max_enemies_beaten)


            print("\tNew Diversity of Subsample: ", np.mean(pdist(pop, metric="euclidean")))
            # Save population
            POP += copy.deepcopy(pop)  # append new population to the whole
            EVALUATIONS += N_GENERATIONS * POP_SIZE  # calculate total amount of evaluations so far
            print("\tEvaluations: ", EVALUATIONS)
            print("----------------------------------------------------------------------------------")
            # Increase iterations
            iterations += 1
        np.savetxt("FITNESS" + str(trial), FITNESS)
        np.savetxt("max_enemies_beaten" + str(trial), np.array(best_performing_array))
        print(f"Total time (minutes): {(time.time() - time_start) / 60:.2f}")
        print("Done!")
