"""
Implementation of multi-objective optimisation using pymoo

Algorithm: SMS-EMOA
Algorithm paper: https://sci-hub.se/https://doi.org/10.1016/j.ejor.2006.08.008
"""
import os

import pymoo.gradient.toolbox as anp
from evoman.environment import Environment
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter

from demo_controller import player_controller
from utils import simulation, verify_solution

N_GENERATIONS = 50
POP_SIZE = 100
ENEMIES = [6, 2, 5, 7, 8]
MODE = "train"  # train or test

n_hidden_neurons = 10

experiment_name = 'cma_v2_test'
solution_file_name = 'cma_v2_best.txt'
os.environ["SDL_VIDEODRIVER"] = "dummy"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=ENEMIES,
                  multiplemode="no",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  logs="off",
                  visuals=False)


class objectives(Problem):

    def __init__(self):
        super().__init__(n_var=265, n_obj=2, n_constr=0, xl=-1, xu=1, type_var=float)

    def _evaluate(self, x, out, *args, **kwargs):
        # Initialize
        dict_enemies = {}
        # Get fitness for each enemy
        for enemy in ENEMIES:
            env.update_parameter('enemies', [enemy])
            dict_enemies[enemy] = []
            for i in range(POP_SIZE):
                dict_enemies[enemy].append(simulation(env, x[i, :], inverted_fitness=True))
        # Stack fitness
        dict_enemies[2578] = []  # , dict_enemies[4] = [], []
        for i in range(POP_SIZE):
            dict_enemies[2578].append(max([dict_enemies[j][i] for j in [2, 5, 7, 8]]))  # Temporarily
            # dict_enemies[4].append(max([dict_enemies[j][i] for j in [4]])) # Temporarily

        out["F"] = anp.column_stack([dict_enemies[6], dict_enemies[2578]])


if __name__ == '__main__':
    problem = objectives()

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
        print(algorithm.n_gen, algorithm.evaluator.n_eval)
        print(1 / algorithm.result().F)

    # obtain the result objective from the algorithm
    res = algorithm.result()
    # calculate a hash to show that all executions end with the same result
    print("hash", res.F.sum())

    res.F = 1 / res.F
    print(res.F)

    for i, x in enumerate(res.X):
        print("***************************")
        print("Point: ", i)
        verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8])

    Scatter().add(res.F, facecolor="none", edgecolor="red").show()

    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, color="red")
    plot.show()
