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
import matplotlib.pyplot as plt

from utils import simulation, verify_solution, init_env

N_GENERATIONS = 10
POP_SIZE = 50
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
                                          'maxiter': N_GENERATIONS})

    while not es.stop():
        X = es.ask()  # Get list of new solutions
        # Initialize fitness and n_enemies lists
        fit, n_enemies = [], []
        # Evaluate each solution in X
        for x in X:
            ft, nen = simulation(env, x)  # evaluate each solution
            fit.append(ft)
            n_enemies.append(nen)
        # print(
        #     f"Generation {es.countiter}: Best fitness: {1 / min(fit):.4f},\t mean: {1 / np.mean(fit):.4f},\t "
        #     f"worst: {max(fit):.2f},\t std: {np.std(fit):.1f},\t n enemies: {max(n_enemies)}")
        # TODO: This is test mutation. Must be removed
        # X = [x + np.random.normal(0, 0.1, len(x)) for x in X]
        # X = [np.random.normal(0, 0, len(x)) for x in X]

        es.tell(X, fit)  # besides for termination only the ranking in fit is used
        print("Best fitness: ", 1 / es.best.f)
        print("Enemies current generation: ", max(n_enemies))
        es.logger.add()  # write data to disc to be plotted
        es.disp()
    print('termination:', es.stop())

    best_solution = es.best.x
    print("\n\n---- BEST ----")
    print(f"Inverted best fitness: {es.best.f}")
    print(f"Inverted best fitness: {simulation(env, best_solution, inverted_fitness=True)}")
    print(f"Original best fitness: {simulation(env, best_solution, inverted_fitness=False)}")

    # save best solution
    np.savetxt(f'{experiment_name}/{solution_file_name}', best_solution)

    verify_solution(env, best_solution)

    es.result_pretty()
    cma.plot()
    plt.show()


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
