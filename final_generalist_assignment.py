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
from pymoo.visualization.scatter import Scatter
from utils import simulation, verify_solution, init_env

TOTAL_ITERATIONS = 2
N_GENERATIONS = 25
POP_SIZE = 20

global ENEMIES
global CLUSTER

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
        """

        # Initialize
        dict_enemies = {}
        # Get fitness for each enemy
        for enemy in self.enemies:
            self.env.update_parameter('enemies', [enemy])

            dict_enemies[enemy] = []
            for individual_id in range(len(x)):
                if self.env.randomini == "no":
                    dict_enemies[enemy].append(simulation(self.env, x[individual_id], inverted_fitness=True))
                else:
                    sims = []
                    for rep_rand in range(0, 5):
                        sims.append(simulation(self.env, x[individual_id], inverted_fitness=True))
                    dict_enemies[enemy].append(np.mean(sims))
        objectives_fitness = {}
        for icl, cl in enumerate(CLUSTER):
            objectives_fitness[f"objective_{icl + 1}"] = [np.max([dict_enemies[enemy_id][ind_id] for enemy_id in cl])
                                                          for ind_id in range(len(x))]

        for ienemy, enemy in enumerate(ENEMIES):
            objectives_fitness[f"objective_{ienemy + 2}"] = dict_enemies[enemy]

        out["F"] = anp.column_stack([objectives_fitness[key] for key in objectives_fitness.keys()])


def plot_pareto_fronts(res):
    """Plot the pareto fronts for each pair of objectives and all 3 objectives"""
    plot = Scatter(labels=["Hard enemies", "Medium Enemies", "Easy enemies"], title="Pareto Front")
    plot.add(res.F, color="red")
    plot.show()

    # for 3 objectives plot each pair of pareto fronts
    # Hard vs Medium
    plot = Scatter(labels=["Hard enemies", "Medium Enemies"], title="Pareto Front")
    plot.add(res.F[:, [0, 1]], color="red")
    plot.show()

    # Hard vs Easy
    plot = Scatter(labels=["Hard enemies", "Easy Enemies"], title="Pareto Front")
    plot.add(res.F[:, [0, 2]], color="red")
    plot.show()

    # Medium vs Easy
    plot = Scatter(labels=["Medium enemies", "Easy Enemies"], title="Pareto Front")
    plot.add(res.F[:, [1, 2]], color="red")
    plot.show()


def main(env: Environment, n_genes: int, population=None, pmut=1, vsigma=1, pcross=1, crossovermode="NN",
         algorithm=None):
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=[1, 2, 3, 4, 5, 6, 7, 8],
        n_objectives=len(ENEMIES) + (len(CLUSTER) > 0)
    )

    if crossovermode == "NN":
        crossover = NNCrossover(prob=pcross)
    elif crossovermode == "SBX":
        crossover = SBX(prob=pcross)
    else:
        raise ValueError("Crossover mode not recognized")

    # TODO: Are we just dropping the new population for the algorithm that was already hashed ???
    if algorithm is None:
        if population is None:
            algorithm = SMSEMOA(pop_size=POP_SIZE, crossover=crossover,
                                mutation=GaussianMutation(prob=pmut, sigma=vsigma))  # , seed=1
        else:
            population = np.array(population)
            algorithm = SMSEMOA(pop_size=POP_SIZE, sampling=population, crossover=crossover,
                                mutation=GaussianMutation(prob=pmut, sigma=vsigma))  # , seed=1
            algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), verbose=False)

    step = 0
    while algorithm.has_next():
        print("\t\t", np.round((step / N_GENERATIONS * 100), 0), "%", end="\r")
        if (step == 1) and (algorithm is None):
            pop = Population(population)
        else:
            pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop)
        algorithm.tell(infills=pop)
        step += 1

    # obtain the result objective from the algorithm
    res = algorithm.result()

    res.F = 1 / res.F

    # max_enemies_beaten = 0
    # best_solutions = []
    # best_not_beaten = []
    best_x, max_enemies_beaten, best_enemies = [], 0, np.zeros(8)
    for i, x in enumerate(res.X):
        enemies_beaten, enemies_not_beaten, enemy_lives = verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                                                                          verbose=True,
                                                                          print_results=False)
        enemy_lives = np.array(enemy_lives)
        if len(enemies_beaten) > max_enemies_beaten:
            max_enemies_beaten = len(enemies_beaten)
            best_x = [x]
            best_enemies = np.where(enemy_lives == 0, 1, 0)
        elif len(enemies_beaten) == max_enemies_beaten:
            best_x.append(x)
            best_enemies += np.where(enemy_lives == 0, 1, 0)

    best_enemies = best_enemies / len(best_x)

    # # save the best solutions to files
    # for i, solution in enumerate(best_solutions):
    #     np.savetxt(f'{experiment_name}/{solution_file_name}_{i}', solution)

    return [i.X for i in algorithm.ask()], best_x, max_enemies_beaten, best_enemies, algorithm


N_REPEATS = 8

if __name__ == '__main__':
    for trial in range(0, N_REPEATS):
        time_start = time.time()
        print("----------------------------------------------------------------------------------")
        # Environment
        env, n_genes = init_env(experiment_name, [1], n_hidden_neurons)
        env.update_parameter('multiplemode', 'no')
        env.update_parameter('level', 2)
        env.update_parameter('randomini', "no")
        # Parameters
        pop = np.random.uniform(-1, 1, size=(100, 265))
        EVALUATIONS = 0
        ENEMIES = np.array([1])
        # N_GENERATIONS = 30
        # POP_SIZE = 50
        # pmut, vsigma = 0.01, 1
        # # Part 1 --> focus on enemies 1 and 7
        # print("\tPopulation 1")
        # CLUSTER = [[1]]
        # ENEMIES = np.array([7])
        # ALL_ENEMIES = CLUSTER[0] + list(ENEMIES)

        # env.update_parameter('enemies', ALL_ENEMIES)
        # pop0 = main(env, n_genes, pmut = pmut, vsigma= vsigma, crossovermode = "SBX")
        # EVALUATIONS = N_GENERATIONS * POP_SIZE

        # # Part 2 --> focus on enemies 4 and 6
        # print("\n\tPopulation 2")
        # CLUSTER = [[4]]
        # ENEMIES = np.array([6])
        # ALL_ENEMIES = CLUSTER[0] + list(ENEMIES)

        # env.update_parameter('enemies', ALL_ENEMIES)
        # pop = main(env, n_genes, pmut = pmut, vsigma= vsigma, crossovermode = "SBX")

        # # Combine populations
        # pop = pop0 + pop
        # EVALUATIONS += N_GENERATIONS * POP_SIZE

        # Save population and population size
        POP = copy.deepcopy(pop)
        popsize_or = copy.deepcopy(len(pop))

        # --------------------------- Iterated Learning/Constrained Led Approach
        # print("Training Round 6", end="\r")
        N_GENERATIONS = 10  #
        POP_SIZE = 20
        pmut, vsigma, pcross = 1, 1, 1
        crossovermode = "NN"
        env.update_parameter('randomini', "no")

        min_n_enemies = 3
        max_n_enemies = 3
        nhistory = 10  # For beaten2
        assert nhistory % 2 == 0, "nhistory must be even"
        weights = 1 / ((np.arange(-(nhistory - 1) / 2, nhistory / 2,
                                  1) ** 2) + 0.5)  # Parabolic weights --> 0.5 displacement due to zero value

        algos = {}
        beaten = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        beaten2 = {1: nhistory * [0], 2: nhistory * [0], 3: nhistory * [0], 4: nhistory * [0], 5: nhistory * [0],
                   6: nhistory * [0], 7: nhistory * [0], 8: nhistory * [0]}
        beaten3 = np.zeros(8)
        best_performing = 0
        BEST_x = ""

        iterations = 0
        FITNESS = np.zeros((800, 301))
        while EVALUATIONS < 50000:
            # env.update_parameter('randomini', "yes")
            # Evaluate population --> beat rates
            for enemy in range(1, 9):
                beaten[enemy] = 0
                beaten2[enemy] = beaten2[enemy][1:] + [0]

            most_beaten, most_x = 0, ""
            for i_x, x in enumerate(POP):
                enemy_beaten = 0
                for enemy in range(1, 9):
                    env.update_parameter('enemies', [enemy])
                    p, e, t = simulation(env, x, verbose=True)
                    f = (100 - e) + np.log(p + 0.001)
                    # Store fitness
                    FITNESS[int(i_x + (enemy - 1) * 100), iterations] = f

                    # Update beaten
                    if (e == 0) and (p > 0):
                        beaten[enemy] += 1
                        beaten2[enemy][-1] += 1
                        enemy_beaten += 1
                    if i_x == (len(POP) - 1):
                        beaten2[enemy][-1] = beaten2[enemy][-1] / popsize_or * 100
                        beaten[enemy] = beaten[enemy] / popsize_or * 100
                if enemy_beaten >= most_beaten:
                    most_beaten = enemy_beaten
                    best_x = copy.deepcopy(x)

            if most_beaten >= best_performing:
                if most_beaten == best_performing:
                    BEST_x = np.loadtxt("BEST_SOLUTION" + str(trial))
                    BEST_x = np.concatenate([BEST_x, best_x], axis=0)
                else:
                    BEST_x = copy.deepcopy(best_x)
                best_performing = copy.deepcopy(most_beaten)
                np.savetxt("BEST_SOLUTION" + str(trial), BEST_x)

            # Update params
            # min_percentage = min(list(beaten.values()))
            # pmut = (1 / (max([0, min_percentage - 10])**1.1 + 0.5)) / (1 / 0.5) * 1 # Exponential decrease if minimum %beaten is 10
            # vsigma = (1 / (max([0, min_percentage -  10])**1.1 + 0.5)) / (1 / 0.5) * 1 # Exponential decrease if minimum %beaten is 10
            # if iterations < 10: # Switch to Neural Net Crossover from SBX
            #    crossovermode = "SBX"
            # else:

            print("NEW ITERATION: ", iterations)
            print("Population Size: ", POP_SIZE)
            print("Mutation Probability: ", pmut)
            print("Crossover Type: ", crossovermode)
            print(f"\tEnemies beaten: {most_beaten}")
            print("\tCurrent Record: ", best_performing)
            print("\tBeaten Percentages Current Population [%] and STD over " + str(nhistory) + " Runs [%]")
            for enemy in range(1, 9):
                print("\t\tEnemy: ", enemy)
                print("\t\t\t" + str(np.round(beaten[enemy], 2)) + "% - " + str(np.std(beaten2[enemy]).round(2)))
            print("\tPopulation Diversity: ", np.mean(pdist(POP, metric="euclidean")))

            # Select population
            idx_pop = np.random.choice(range(len(POP)), size=POP_SIZE, replace=False)
            pop = [POP[idx] for idx in idx_pop]
            POP = [POP[idx] for idx in range(len(POP)) if idx not in idx_pop]

            print("\tOld Diversity of Subsample: ", np.mean(pdist(pop, metric="euclidean")))

            # Set enemies
            beaten_vals = beaten3
            if sum(beaten_vals) == 0:
                probs = np.ones(8) / 8
            else:
                probs = sum(beaten_vals) / np.where(beaten_vals == 0, 0.01, beaten_vals)
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
                algos = {algorithm_hash: algorithm}

            if max_enemies_beaten > best_performing:
                best_performing = max_enemies_beaten
                BEST_x = best_x[0]
                np.savetxt("BEST_SOLUTION" + str(trial), BEST_x)
            elif max_enemies_beaten == best_performing:
                BEST_x = np.loadtxt("BEST_SOLUTION" + str(trial))
                BEST_x = np.concatenate([BEST_x, best_x[0]], axis=0)
                np.savetxt("BEST_SOLUTION" + str(trial), BEST_x)

            ## Save algo
            beaten3 += best_enemies
            print(beaten3)

            print("\tNew Diversity of Subsample: ", np.mean(pdist(pop, metric="euclidean")))
            # Save population
            POP += copy.deepcopy(pop)
            EVALUATIONS += N_GENERATIONS * POP_SIZE
            print("\tEvaluations: ", EVALUATIONS)
            print("----------------------------------------------------------------------------------")
            # Increase iterations
            iterations += 1
        np.savetxt("FITNESS" + str(trial), FITNESS)
        print(f"Total time (minutes): {(time.time() - time_start) / 60:.2f}")
        print("Done!")
