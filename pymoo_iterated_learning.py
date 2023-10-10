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

from nn_crossover import NNCrossover
from matplotlib import pyplot as plt
import numpy as np
import pymoo.gradient.toolbox as anp
from evoman.environment import Environment
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.util.running_metric import RunningMetricAnimation
from pymoo.visualization.scatter import Scatter

from utils import simulation, verify_solution, init_env

np.random.seed(1)
TOTAL_ITERATIONS = 100
N_GENERATIONS = 25
POP_SIZE = 50

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
        #        weighted average is another option, but then we have another problem of how to weight the enemies
        """

        # Initialize
        dict_enemies = {}
        # Get fitness for each enemy
        for enemy in self.enemies:
            self.env.update_parameter('enemies', [enemy])

            dict_enemies[enemy] = []
            for individual_id in range(POP_SIZE):
                dict_enemies[enemy].append(simulation(self.env, x[individual_id], inverted_fitness=True))

        # Return fitness outputs for enemies
        # objectives_fitness = {
        #     "objective_1": [np.max([dict_enemies[enemy_id][ind_id] for enemy_id in [1, 6]]) for ind_id in
        #                     range(POP_SIZE)],
        #     "objective_2": [np.max([dict_enemies[enemy_id][ind_id] for enemy_id in [2, 5, 8]]) for ind_id in
        #                     range(POP_SIZE)],
        #     "objective_3": [np.max([dict_enemies[enemy_id][ind_id] for enemy_id in [3, 4, 7]]) for ind_id in
        #                     range(POP_SIZE)],
        # }
        objectives_fitness = {
            "objective_1": [np.max([dict_enemies[enemy_id][ind_id] for enemy_id in CLUSTER]) for ind_id in
                            range(POP_SIZE)]
        }

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


def main(env: Environment, n_genes: int, population=None):
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=[1, 2, 3, 4, 5, 6, 7, 8],
        n_objectives=len(ENEMIES) + (len(CLUSTER) > 0)
    )

    if population is None:
        algorithm = SMSEMOA(pop_size=POP_SIZE, seed=1, crossover=NNCrossover())
    else:
        population = np.array(population)
        algorithm = SMSEMOA(pop_size=POP_SIZE, seed=1, sampling=population, crossover=NNCrossover())
    # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
    algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), verbose=False)

    while algorithm.has_next():
        pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop)
        algorithm.tell(infills=pop)

    # obtain the result objective from the algorithm
    res = algorithm.result()

    res.F = 1 / res.F

    max_enemies_beaten = 0
    best_solutions = []
    best_not_beaten = []
    env.update_parameter('level', 2)
    for i, x in enumerate(res.X):
        enemies_beaten, enemies_not_beaten, _ = verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8], verbose=True,
                                                                print_results=False)
        if len(enemies_beaten) > max_enemies_beaten:
            max_enemies_beaten = len(enemies_beaten)
            best_solutions = [x]  # reset the list because we found a better performing solution
            best_not_beaten = [enemies_not_beaten]
        elif len(enemies_beaten) == max_enemies_beaten:
            best_solutions.append(x)  # add to the list the solution that beats the same number of enemies
            best_not_beaten.append(enemies_not_beaten)

    # # save the best solutions to files
    # for i, solution in enumerate(best_solutions):
    #     np.savetxt(f'{experiment_name}/{solution_file_name}_{i}', solution)

    return [i.x for i in algorithm.ask()], best_not_beaten, best_solutions


if __name__ == '__main__':
    time_start = time.time()

    CLUSTER = [1]
    ENEMIES = np.array([2, 3, 4, 5, 6, 7, 8])

    print("Running...")
    env: Environment
    env, n_genes = init_env(experiment_name, ENEMIES, n_hidden_neurons)
    env.update_parameter('multiplemode', 'no')
    env.update_parameter('level', 2)
    env.update_parameter('randomini', "yes")  # random initial position. This is good for diversity
    # TODO: If we find solution that beats 6 or more enemies, we can set randomini to "no" to ensure that they are
    # proper solutions. Also we may vary this parameter during the training process

    pop, best_not_beaten, best_solutions = main(env, n_genes)
    evaluations = POP_SIZE * N_GENERATIONS
    randomini_prob = 0.5

    i = 1
    while ENEMIES.size != 0:
        print("Iteration: ", i)
        # Set enemies
        CLUSTER = [enemy for enemy in range(1, 9) if enemy not in best_not_beaten[0]]
        if not CLUSTER:
            CLUSTER = [np.random.choice(best_not_beaten[0])]
        if len(CLUSTER) >= 7:
            env.update_parameter('randomini', "no")
        else:
            env.update_parameter('randomini', "yes" if np.random.random() < randomini_prob else "no")
        if len(CLUSTER) == 8:
            # there is a solution that beats all enemies
            # save the best solutions to files
            for i, solution in enumerate(best_solutions):
                np.savetxt(f'{experiment_name}/{solution_file_name}_{i}', solution)

        print(f"Cluster (currently beating): {CLUSTER}")  # best performing solution beats these enemies
        ENEMIES = [enemy for enemy in best_not_beaten[0] if enemy not in CLUSTER]
        ENEMIES = np.random.choice(ENEMIES, np.random.choice(np.arange(1, len(ENEMIES) + 1)), replace=False)
        print(f"Enemies (training to beat): {ENEMIES}")  # the population is training to beat these enemies
        print(f"Random initial position: {env.randomini}")
        print(f"Enemies beaten in current population: {len(CLUSTER)}/8")
        print(f"Objective Function Evaluations: {evaluations}")

        # Update number of evaluations
        evaluations += POP_SIZE * N_GENERATIONS
        pop, best_not_beaten, best_solutions = main(env, n_genes, population=pop)

        i += 1
        if i > TOTAL_ITERATIONS:
            break
        print("----")

    print(f"Total time (minutes): {(time.time() - time_start) / 60:.2f}")
    print("Done!")
