"""
Implementation of multi-objective optimisation using pymoo

https://pymoo.org/algorithms/list.html


Algorithm: SMS-EMOA
Algorithm paper: https://sci-hub.se/https://doi.org/10.1016/j.ejor.2006.08.008
Docs link: https://pymoo.org/algorithms/moo/sms.html
"""
import copy
import time
import uuid
from typing import Optional

import numpy as np
import pandas as pd
from evoman.environment import Environment
from scipy.spatial.distance import pdist

import pymoo.gradient.toolbox as anp
from nn_crossover import NNCrossover
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.config import Config
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GaussianMutation
from utils import simulation, verify_solution, init_env, initialise_script

# Settings
N_REPEATS = 1
MAX_EVALUATIONS = 50_000 + 200  # increase by 200 iterations more because we do not log the last iteration
N_GENERATIONS = 10
POP_SIZE = 20  # Subpopulation
WHOLE_POP_SIZE = 100  # whole population
enemies_list = [1, 2, 3, 4, 5, 6, 7, 8]

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
experiment_name = 'final_generalist_assignment'
initialise_script(experiment_name=experiment_name, clean_folder=False)
Config.warnings['not_compiled'] = False


class objectives(Problem):
    enemies: list[int]
    env: Environment
    last_iteration_objectives_fitness: Optional[dict] = None

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

        # Store the fitness for the last iteration
        self.last_iteration_objectives_fitness = objectives_fitness

        # Get the fitness for the whole population and all objectives
        out["F"] = anp.column_stack([objectives_fitness[key] for key in objectives_fitness.keys()])


def main(env: Environment, n_genes: int, population=None, pmut=1, vsigma=1, pcross=1, crossovermode="NN",
         algorithm=None, current_iteration=None):
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=ALL_ENEMIES,
        n_objectives=len(ALL_ENEMIES)
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
    algo_datastore = []  # list with: (n_iter, n_gens, ind_id, mean_obj, obj_1, obj_2, obj_3)
    while algorithm.has_next():
        print("\t\t", np.round((step / N_GENERATIONS * 100), 0), "%", end="\r")
        if (step == 0) and (algo_label is True):  # If we are in the first step + we have an algorithm
            pop = Population(population)
        else:
            pop = algorithm.ask()
        # Evaluate the individuals
        algorithm.evaluator.eval(problem, pop)
        # Store the data
        fvalues = np.array([val for val in problem.last_iteration_objectives_fitness.values()])
        for ind_id in range(len(pop)):
            individual_objective_fitness_list = 1 / fvalues[:, ind_id]  # for maximization
            fitness_mean = (individual_objective_fitness_list).mean()

            algo_datastore.append(
                [current_iteration, algorithm.n_gen, ind_id, fitness_mean] +
                individual_objective_fitness_list.tolist()
            )
        # Tell the algorithm the fitness of the individuals
        algorithm.tell(infills=pop)
        # Increase step
        step += 1

    # --- Check results
    # Initialize
    best_x, max_enemies_beaten, best_enemies = [], 0, np.zeros(8)

    # Obtain the result objective from the algorithm
    res = algorithm.result()

    # Obtain the best solutions in terms of enemies beaten --> used for probabilities
    for i, x in enumerate(res.X):
        # Get enemy lives
        enemies_beaten, enemies_not_beaten, enemy_lives = verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                                                                          verbose=True,
                                                                          print_results=False)
        # Convert enemy lives to array
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

    return [i.X for i in algorithm.ask()], best_x, max_enemies_beaten, best_enemies, algorithm, algo_datastore


if __name__ == '__main__':
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Staring script\n")
    time_start = time.time()
    for trial in range(0, N_REPEATS):
        # --------------------------- Overall Initialization
        print(f"Staring repeat {trial + 1}/{N_REPEATS}")
        # Create unique identifier for the trial
        trial_uuid = uuid.uuid4()
        # Number of evaluations and enemies
        EVALUATIONS, ENEMIES = 0, np.array([1])
        # Environment
        env, n_genes = init_env(experiment_name, [1], n_hidden_neurons)
        env.update_parameter('multiplemode', 'no')
        env.update_parameter('randomini', 'no')
        env.update_parameter("level", 2)
        print("----------------------------------------------------------------------------------")
        # --------------------------- Initialize population
        # Sample random initial population
        pop = np.random.uniform(-1, 1, size=(WHOLE_POP_SIZE, n_genes))
        # Save population and population size
        POP, popsize_or = copy.deepcopy(pop), copy.deepcopy(len(pop))
        # --------------------------- Iterated Learning/Constrained Led Approach
        assert nhistory % 2 == 0, "nhistory must be even"
        # Initialize Constrained Led Approach
        iterations = 0  # Number of iterations
        population_max_enemies_beaten = 0  # Maximum number of enemies beaten in current population
        current_record_max_enemies_beaten = 0  # Most enemies beaten by a single individual ever
        # format: (n_iters, n_evals, max_enemies_beaten)
        best_performing_array = []  # Most enemies beaten by a single individual in current population over time
        BEST_x = ""  # Best individual --> list later on
        algos = {}  # Cache algorithms

        # data store format:
        # (n_iter, n_gens, ind_id, mean_obj, obj_1, obj_2, obj_3)
        trial_datastore = []  # Data store in format
        # (n_iter, n_evals, ind_id, mean_fitness, f1,f2, f3, f4, f5, f6, f7, f8)
        trial_datastore_secondary = []  # Data store in format

        # Number of enemies beaten in current population
        beaten = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        # Number of enemies beaten in current population over time nhistory
        beaten2 = {1: nhistory * [0], 2: nhistory * [0], 3: nhistory * [0], 4: nhistory * [0], 5: nhistory * [0],
                   6: nhistory * [0], 7: nhistory * [0],
                   8: nhistory * [0]}
        # Sum of average number of enemies beaten in current Pareto front for the most generalizable enemies
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
                n_enemies_beaten_by_ind = 0
                f_per_enemy = []
                for enemy in range(1, 9):
                    # Simulate
                    env.update_parameter('enemies', [enemy])
                    p, e, t = simulation(env, x, verbose=True)
                    f = (100 - e) + np.log(p + 0.001)
                    f = max(f, 0.00001)
                    f_per_enemy.append(f)
                    # Update beaten
                    if (e == 0) and (p > 0):
                        beaten[enemy] += 1
                        beaten2[enemy][-1] += 1
                        n_enemies_beaten_by_ind += 1
                    if i_x == (len(POP) - 1):  # Last one --> Normalize
                        beaten2[enemy][-1] = beaten2[enemy][-1] / popsize_or * 100
                        beaten[enemy] = beaten[enemy] / popsize_or * 100
                # Reset
                if n_enemies_beaten_by_ind > most_beaten:
                    most_beaten = copy.deepcopy(n_enemies_beaten_by_ind)
                    best_x = copy.deepcopy(x)
                elif (n_enemies_beaten_by_ind == most_beaten):
                    if (most_beaten != 0) or (i_x != 0):  # if best_x exists
                        best_x = np.vstack((best_x, x))
                    else:  # best_x does not exist yet
                        best_x = copy.deepcopy(x)

                # Add fitness to datastore
                trial_datastore_secondary.append([iterations, EVALUATIONS, i_x, np.mean(f_per_enemy)] + f_per_enemy)

            # ---- Update best performing
            if most_beaten >= current_record_max_enemies_beaten:
                if iterations == 0:  # BEST does not exist yet
                    BEST = copy.deepcopy(best_x)
                elif (most_beaten == current_record_max_enemies_beaten):
                    BEST = np.vstack((BEST, best_x))
                else:  # New Best
                    BEST = copy.deepcopy(best_x)
                # Update
                current_record_max_enemies_beaten = copy.deepcopy(most_beaten)

            if iterations == 0:  # Store initial value of population
                entry = (iterations, EVALUATIONS, most_beaten)
                best_performing_array.append(entry)
            elif most_beaten > population_max_enemies_beaten:  # Substitute best performing
                entry = (iterations, EVALUATIONS, most_beaten)
                best_performing_array[-1] = entry  # Because this lacks one behind next update

            # --- Print some settings
            print(f"Iteration: {iterations}; Evaluation: {EVALUATIONS}")
            print("Population Size: ", POP_SIZE)
            print("Mutation Probability: ", pmut)
            print("Crossover Type: ", crossovermode)
            print("\tCurrent Record: ", current_record_max_enemies_beaten)
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

            # Set enemies according to probabilities
            beaten_vals = beaten3[0:len(enemies_list)]  # TODO: verify this bit when we run for 7 enemies
            if sum(beaten_vals) == 0:
                probs = np.ones(len(enemies_list)) / len(enemies_list)
            else:
                min_val = np.min(np.where(beaten_vals == 0, 999, beaten_vals))
                probs = sum(beaten_vals) / np.where(beaten_vals == 0, min_val, beaten_vals)
                probs = probs / sum(probs)

            # Choose 3 enemies with inverse probabilities.
            # Enemies on which the population performs poorly will be chosen with higher probability
            n_enemies = np.random.choice([min_n_enemies, max_n_enemies])
            opponents = np.random.choice(enemies_list, p=probs, size=n_enemies, replace=False)
            CLUSTER = [[opponents[0]]]
            ENEMIES = np.array([opponents[i] for i in range(1, n_enemies)])

            try:
                ALL_ENEMIES = [enemy for cl in CLUSTER for enemy in cl] + list(ENEMIES)
            except TypeError:
                ALL_ENEMIES = [enemy for enemy in CLUSTER] + [ENEMIES]

            print("\tTraining on: ")
            print(f"\t\tEnemy 1: {CLUSTER}")
            print(f"\t\tEnemy 2,3: {ENEMIES}")

            # Get 0
            if iterations == 0:
                for i_x, x in enumerate(pop):
                    fs = []
                    for enemy in ALL_ENEMIES:
                        env.update_parameter('enemies', [enemy])
                        f = simulation(env, x, inverted_fitness=False)
                        fs.append(max(f, 0.00001))
                    trial_datastore.append([0, 0, i_x, np.mean(fs)] + [f for f in fs])

            # Create unique identifier for the algorithm instance
            ALL_ENEMIES.sort()  # sort for hashing reproducibility
            algorithm_hash = "-".join([str(i) for i in ALL_ENEMIES])

            # Check if algorithm was already initialised and run
            if algorithm_hash in algos.keys():
                pop, best_x, population_max_enemies_beaten, best_enemies, algorithm, algo_datastore = main(
                    env, n_genes,
                    population=pop,
                    pmut=pmut,
                    vsigma=vsigma,
                    pcross=pcross,
                    crossovermode=crossovermode,
                    algorithm=algos[
                        algorithm_hash],
                    current_iteration=iterations)
            else:
                pop, best_x, population_max_enemies_beaten, best_enemies, algorithm, algo_datastore = main(
                    env, n_genes,
                    population=pop,
                    pmut=pmut,
                    vsigma=vsigma,
                    pcross=pcross,
                    crossovermode=crossovermode,
                    current_iteration=iterations)
            # Cache algorithm instance
            algos = {algorithm_hash: algorithm}

            # Increase beaten3
            beaten3 += best_enemies
            print("Beat Rates of Most Generalizable Enemies: ", beaten3)

            # Check for best performing in Pareto front --> before we saved these x-values, but now we don't because of speed considerations
            # We use an archived based approach, so this is allowed. However, it also means that these solution might not end up in our plots
            # , so the might differ because we have some spacing (Ngeneration * Popsize) between storage of fitness. But we cannot store all fitness values
            if population_max_enemies_beaten > current_record_max_enemies_beaten:  # New best
                current_record_max_enemies_beaten = copy.deepcopy(population_max_enemies_beaten)
                BEST = copy.deepcopy(best_x)
            elif population_max_enemies_beaten == current_record_max_enemies_beaten:  # Equal best
                BEST = np.vstack((BEST, best_x))
            # Append to best_performing
            # Append because of after ... evaluations after previous update
            best_performing_array.append((iterations, EVALUATIONS, population_max_enemies_beaten))

            print("\tNew Diversity of Subsample: ", np.mean(pdist(pop, metric="euclidean")))
            # Save population
            trial_datastore.extend(algo_datastore)
            POP += copy.deepcopy(pop)  # append new population to the whole
            EVALUATIONS += N_GENERATIONS * POP_SIZE  # calculate total amount of evaluations so far
            print(f"\tEvaluations: {EVALUATIONS}/{MAX_EVALUATIONS}")
            print("----------------------------------------------------------------------------------")
            # Increase iterations
            iterations += 1

        # (n_iter, n_gens, ind_id, mean_obj, obj_1, obj_2, obj_3)
        df = pd.DataFrame(
            trial_datastore,
            columns=("n_iter", "n_gens", "ind_id", "mean_obj", "obj_1", "obj_2", "obj_3")
        )
        df.to_csv(
            f"{experiment_name}/dynamic_objectives_{current_record_max_enemies_beaten}_{trial_uuid}.csv",
            index=False
        )
        # ("n_iter", "n_evals", "ind_id", "mean_fitness", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8")
        # TODO: remove f8 when we run for 7 enemies
        df = pd.DataFrame(
            trial_datastore_secondary,
            columns=("n_iter", "n_evals", "ind_id", "mean_fitness", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8")
        )
        df.to_csv(
            f"{experiment_name}/dynamic_objectives_secondary_{current_record_max_enemies_beaten}_{trial_uuid}.csv",
            index=False
        )
        df = pd.DataFrame(
            best_performing_array,
            columns=("n_iter", "n_evals", "max_enemies_beaten")
        )
        df.to_csv(
            f"{experiment_name}/max_enemies_beaten_{current_record_max_enemies_beaten}_{trial_uuid}.csv",
            index=False
        )
        del df  # free memory
        # find the best solution in the BEST. If there are multiple solutions with the same number of enemies beaten,
        # the one with the highest player life is chosen
        most_enemies_beaten = 0
        most_player_live = 0
        win_id = 0
        for idx, solution in enumerate(BEST):
            enemies_beaten, _, _, player_lives, _ = verify_solution(env, best_solution=solution, vv=True,
                                                                    print_results=False)
            if len(enemies_beaten) > most_enemies_beaten:
                most_enemies_beaten = len(enemies_beaten)
                most_player_live = np.sum(player_lives)
                win_id = idx
            elif len(enemies_beaten) == most_enemies_beaten:
                # check if the player life is higher
                if np.sum(player_lives) > most_player_live:
                    most_player_live = np.sum(player_lives)
                    win_id = idx
        the_best_of_the_best_genotype = BEST[win_id]
        np.savetxt(f"{experiment_name}/best_solutions_{current_record_max_enemies_beaten}_{trial_uuid}.txt",
                   the_best_of_the_best_genotype)
        print(f"Total time (minutes cumulative): {(time.time() - time_start) / 60:.2f}")
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ---- Done!\n")
