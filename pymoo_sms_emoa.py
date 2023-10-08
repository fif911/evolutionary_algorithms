"""
Implementation of multi-objective optimisation using pymoo

https://pymoo.org/algorithms/list.html


Algorithm: SMS-EMOA
Algorithm paper: https://sci-hub.se/https://doi.org/10.1016/j.ejor.2006.08.008
"""
import os

import numpy as np
import pymoo.gradient.toolbox as anp
from evoman.environment import Environment
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter

from utils import simulation, verify_solution, init_env

N_GENERATIONS = 50
POP_SIZE = 100
ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]

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
        objectives_fitness = {
            # TODO: check correctness
            "objective_1": [max([dict_enemies[enemy_id][ind_id] for enemy_id in [1, 6]]) for ind_id in range(POP_SIZE)],
            "objective_2": [max([dict_enemies[enemy_id][ind_id] for enemy_id in [2, 5, 8]]) for ind_id in range(POP_SIZE)],
            "objective_3": [max([dict_enemies[enemy_id][ind_id] for enemy_id in [3, 4, 7]]) for ind_id in range(POP_SIZE)],
        }

        out["F"] = anp.column_stack([objectives_fitness[key] for key in objectives_fitness.keys()])
        # dict_enemies[2578] = []  # , dict_enemies[4] = [], []
        # for i in range(POP_SIZE):
        #     dict_enemies[2578].append(max([dict_enemies[j][i] for j in [2, 5, 7, 8]]))  # Temporarily
        #     # dict_enemies[4].append(max([dict_enemies[j][i] for j in [4]])) # Temporarily


def main(env: Environment, n_genes: int):
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=ENEMIES,
        n_objectives=3
    )

    algorithm = SMSEMOA(pop_size=POP_SIZE)  # https://sci-hub.se/https://doi.org/10.1016/j.ejor.2006.08.008

    # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
    algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), verbose=False)
    # until the algorithm has no terminated
    while algorithm.has_next():
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        algorithm.evaluator.eval(problem, pop)
        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)
        # do same more things, printing, logging, storing or even modifying the algorithm object
        # print(algorithm.n_gen, algorithm.evaluator.n_eval)
        print(f"Generation: {algorithm.n_gen}")
        print(f"Best individual fitness: {', '.join([f'{_:.2f}' for _ in algorithm.result().F[0]])}")

    # obtain the result objective from the algorithm
    res = algorithm.result()
    print(f"Individuals evaluated: {algorithm.evaluator.n_eval}")

    res.F = 1 / res.F
    print(res.F)

    for i, x in enumerate(res.X):
        print(f"------ Solution {i + 1} -----")
        verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8])

    # Scatter().add(res.F, facecolor="none", edgecolor="red").show()

    plot = Scatter(labels=["Hard enemies", "Medium Enemies", "Easy enemies"], title="Pareto Front")
    # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()


if __name__ == '__main__':
    print("Running pymoo_sms_emoa.py")
    env, n_genes = init_env(experiment_name, ENEMIES, n_hidden_neurons)
    env.update_parameter('multiplemode', 'no')

    main(env, n_genes)

    print("Done!")
