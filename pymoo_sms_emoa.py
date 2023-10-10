"""
Implementation of multi-objective optimisation using pymoo

https://pymoo.org/algorithms/list.html


Algorithm: SMS-EMOA
Algorithm paper: https://sci-hub.se/https://doi.org/10.1016/j.ejor.2006.08.008
Docs link: https://pymoo.org/algorithms/moo/sms.html

Algorithm: AGE-MOEA
Algorithm paper: https://sci-hub.se/10.1145/3321707.3321839
Docs link: https://pymoo.org/algorithms/moo/age.html#nb-agemoea

"""
import time

import numpy as np
import pymoo.gradient.toolbox as anp
from evoman.environment import Environment
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.visualization.scatter import Scatter

from nn_crossover import NNCrossover
from utils import simulation, verify_solution, init_env, run_pymoo_algorithm, initialise_script

N_GENERATIONS_LEVEL_1 = 30
N_GENERATIONS_LEVEL_2 = 30
POP_SIZE = 20
ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]

n_hidden_neurons = 10

experiment_name = 'pymoo_sms_emoa'
solution_file_name = 'pymoo_sms_emoa_best'

initialise_script(experiment_name=experiment_name)


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
        objectives_fitness = {
            "objective_hard": [np.mean([dict_enemies[enemy_id][ind_id] for enemy_id in [1, 6]]) for ind_id in
                               range(POP_SIZE)],
            "objective_medium": [np.mean([dict_enemies[enemy_id][ind_id] for enemy_id in [2, 5, 8]]) for ind_id in
                                 range(POP_SIZE)],
            "objective_easy": [np.mean([dict_enemies[enemy_id][ind_id] for enemy_id in [3, 4, 7]]) for ind_id in
                               range(POP_SIZE)],
        }

        out["F"] = anp.column_stack([objectives_fitness[key] for key in objectives_fitness.keys()])


def plot_pareto_fronts(res, best_solutions_idx: list[int]):
    """Plot the pareto fronts for each pair of objectives and all 3 objectives"""
    print(f"Plotting {res.F.shape[0]} solutions")

    plot = Scatter(labels=["Hard enemies", "Medium Enemies", "Easy enemies"], title="Pareto Front")
    plot.add(res.F, color="red")
    plot.add(res.F[best_solutions_idx], color="blue", s=80, label="Best solutions")
    plot.show()

    # for 3 objectives plot each pair of pareto fronts
    # Hard vs Medium
    plot = Scatter(labels=["Hard enemies", "Medium Enemies"], title="Pareto Front")
    plot.add(res.F[:, [0, 1]], color="red")
    plot.add(res.F[:, [0, 1]][best_solutions_idx], color="blue", s=80, label="Best solutions")
    plot.show()

    # Hard vs Easy
    plot = Scatter(labels=["Hard enemies", "Easy Enemies"], title="Pareto Front")
    plot.add(res.F[:, [0, 2]], color="red")
    plot.add(res.F[:, [0, 2]][best_solutions_idx], color="blue", s=80, label="Best solutions")
    plot.show()

    # Medium vs Easy
    plot = Scatter(labels=["Medium enemies", "Easy Enemies"], title="Pareto Front")
    plot.add(res.F[:, [1, 2]], color="red")
    plot.add(res.F[:, [1, 2]][best_solutions_idx], color="blue", s=80, label="Best solutions")
    plot.show()


def main(env: Environment, n_genes: int):
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=ENEMIES,
        n_objectives=3
    )

    if N_GENERATIONS_LEVEL_1:  # skip level 1 if it is 0
        print("Setting the random initialisation position to Yes")
        env.update_parameter("randomini", "yes")
        algorithm = SMSEMOA(pop_size=POP_SIZE, )
        algorithm.setup(problem, termination=('n_gen', N_GENERATIONS_LEVEL_1), verbose=False)

        algorithm = run_pymoo_algorithm(algorithm, problem, postfix="_level_1")
        # get the best individuals from results and population
        # and pass further to the next level
        _population = list(algorithm.result().X)
        _population.extend(list(algorithm.pop.get("X")))
        _population = _population[:POP_SIZE]
        next_population = np.array(_population)
        first_algorithm_evaluations = algorithm.evaluator.n_eval
    else:
        next_population = FloatRandomSampling()
        first_algorithm_evaluations = 0

    print("Setting the random initialisation position to No")
    env.update_parameter("randomini", 'no')
    algorithm = SMSEMOA(pop_size=POP_SIZE, sampling=next_population, crossover=NNCrossover())
    algorithm.setup(problem, termination=('n_gen', N_GENERATIONS_LEVEL_2), verbose=False)

    algorithm = run_pymoo_algorithm(algorithm, problem, postfix="_level_2")

    # obtain the result objective from the algorithm
    res = algorithm.result()

    res.F = 1 / res.F
    print(res.F)

    max_enemies_beaten = 0
    best_solutions = []
    best_solutions_idx = []
    for i, x in enumerate(res.X):
        print(f"------ Solution {i + 1} -----")
        enemies_beaten = verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8])
        if enemies_beaten > max_enemies_beaten:
            max_enemies_beaten = enemies_beaten
            best_solutions = [x]  # reset the list because we found a better performing solution
            best_solutions_idx = [i]
        elif enemies_beaten == max_enemies_beaten:
            best_solutions.append(x)  # add to the list the solution that beats the same number of enemies
            best_solutions_idx.append(i)

    print(f"Most enemies beaten: {max_enemies_beaten}; Number of these solutions: {len(best_solutions)}")
    print(f"Individuals evaluated: {algorithm.evaluator.n_eval + first_algorithm_evaluations}")

    # save the best solutions to files
    for i, solution in enumerate(best_solutions):
        np.savetxt(f'{experiment_name}/{solution_file_name}_{i}.txt', solution)

    plot_pareto_fronts(res, best_solutions_idx)


if __name__ == '__main__':
    time_start = time.time()
    print("Running pymoo_sms_emoa.py")
    env, n_genes = init_env(experiment_name, ENEMIES, n_hidden_neurons)
    env.update_parameter('multiplemode', 'no')

    pop = main(env, n_genes)

    print(f"Total time (minutes): {(time.time() - time_start) / 60:.2f}")
    print("Done!")
