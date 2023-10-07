"""
Install cma package:
pipenv install cma
or
pip install cma
"""

import os
import time

import cma
import numpy as np

from demo_controller import player_controller
from evoman.environment import Environment

from utils import simulation, verify_solution

N_GENERATIONS = 50
POP_SIZE = 50
ENEMIES = [6]
MODE = "train"  # train or test

n_hidden_neurons = 10

experiment_name = 'cma_v2_test'
solution_file_name = 'cma_v2_best.txt'
os.environ["SDL_VIDEODRIVER"] = "dummy"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def solution_search(env):
    n_genes = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    es = cma.CMAEvolutionStrategy(n_genes * [0], 0.8,
                                  inopts={'bounds': [-1, 1],
                                          'popsize': POP_SIZE,
                                          'maxiter': N_GENERATIONS})

    while not es.stop():
        X = es.ask()  # get list of new solutions
        fit = [simulation(env, x) for x in X]  # evaluate each solution
        print(
            f"Generation {es.countiter}: Best fitness: {min(fit):.4f},\t mean: {np.mean(fit):.4f},\t "
            f"worst: {max(fit):.2f},\t std: {np.std(fit):.1f}")

        # TODO: This is test mutation. Must be removed
        # X = [x + np.random.normal(0, 0.1, len(x)) for x in X]
        # X = [np.random.normal(0, 0, len(x)) for x in X]

        es.tell(X, fit)  # besides for termination only the ranking in fit is used
    print('termination:', es.stop())

    best_solution = es.best.x
    print("\n\n---- BEST ----")
    print(f"Inverted best fitness: {es.best.f}")
    print(f"Inverted best fitness: {simulation(env, best_solution, inverted_fitness=True)}")
    print(f"Original best fitness: {simulation(env, best_solution, inverted_fitness=False)}")

    # save best solution
    np.savetxt(f'{experiment_name}/{solution_file_name}', best_solution)

    verify_solution(env, best_solution)


if __name__ == "__main__":
    env = Environment(experiment_name=experiment_name,
                      enemies=ENEMIES,
                      multiplemode="yes" if len(ENEMIES) > 1 else "no",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off",
                      visuals=False)

    if MODE == "train":
        time_start = time.time()
        solution_search(env)
        print("Done!")
        # time in minutes
        print(f"Total time: {(time.time() - time_start) / 60:.2f} minutes")
    elif MODE == "test":
        best_loaded_solution = np.loadtxt(f'{experiment_name}/{solution_file_name}')
        verify_solution(env, best_loaded_solution)
    else:
        raise ValueError(f"MODE {MODE} not supported")
