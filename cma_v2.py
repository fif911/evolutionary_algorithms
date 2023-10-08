"""
Install cma package:
pipenv install cma
or
pip install cma

For CMA recommended number of evaluations is 100 * n_genes which bring us to 265 * 100 = 26,500 evaluations
We can do 100,000 in 3 hours, so this is still ok
"""

import os
import time

import cma
import numpy as np

from utils import simulation, verify_solution, init_env

N_GENERATIONS = 50
POP_SIZE = 50
MAX_EVALUATIONS = 100_000

ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]
MODE = "train"  # train or test

n_hidden_neurons = 10

experiment_name = 'cma_v2_test'
solution_file_name = 'cma_v2_best.txt'
os.environ["SDL_VIDEODRIVER"] = "dummy"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def solution_search(env, n_genes):
    es = cma.CMAEvolutionStrategy(n_genes * [0], 0.8,
                                  inopts={'bounds': [-1, 1],
                                          'popsize': POP_SIZE,
                                          # 'maxiter': N_GENERATIONS,
                                          'maxfevals': MAX_EVALUATIONS,
                                          })

    while not es.stop():
        X = es.ask()  # get list of new solutions
        fit = [simulation(env, x) for x in X]  # evaluate each solution
        print(
            f"Generation {es.countiter}:\t FEvals: {es.countevals},"
            f"\t\t Best fitness: {min(fit):.4f},\t mean: {np.mean(fit):.4f},\t "
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
    env, n_genes = init_env(experiment_name, ENEMIES, n_hidden_neurons)

    if MODE == "train":
        time_start = time.time()
        solution_search(env, n_genes)
        print("Done!")
        # time in minutes
        print(f"Total time: {(time.time() - time_start) / 60:.2f} minutes")
    elif MODE == "test":
        best_loaded_solution = np.loadtxt(f'{experiment_name}/{solution_file_name}')
        verify_solution(env, best_loaded_solution)
    else:
        raise ValueError(f"MODE {MODE} not supported")
