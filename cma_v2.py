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

N_GENERATIONS = 30
POP_SIZE = 50
ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]
MODE = "test"  # train or test

n_hidden_neurons = 10

experiment_name = 'cma_v2_test'
solution_file_name = 'cma_v2_best.txt'
os.environ["SDL_VIDEODRIVER"] = "dummy"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


def simulation(env, xm: np.ndarray, pure_fitness=False, return_enemies=False):
    """Run one episode and return the fitness

    pure_fitness: if True, return the fitness as is, otherwise return the inverse of the fitness for minimization problem
    return_enemies: if True, return the player life, enemy life and time
    """
    f, p, e, t = env.play(pcont=xm)
    if pure_fitness:
        return f
    if return_enemies:
        return p, e, t

    fitness = 0.9 * (100 - e) + 0.1 * p - np.log(t)
    if fitness <= 0:
        fitness = 0.00001

    return 1 / fitness


def verify_solution(env, best_solution):
    enemies_beaten = 0
    env.update_parameter("multiplemode", "no")
    for enemy in ENEMIES:

        env.update_parameter('enemies', [enemy])
        p, e, t = simulation(env, best_solution, return_enemies=True)
        enemy_beaten = e == 0 and p > 0
        print(f"Enemy {enemy};\tPlayer Life: {p}, Enemy Life: {e}, in {t} seconds. \tWon: {enemy_beaten}")
        if enemy_beaten:
            enemies_beaten += 1
    print(f"Enemies beaten: {enemies_beaten}/{len(ENEMIES)}")


def solution_search(env):
    n_genes = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    es = cma.CMAEvolutionStrategy(n_genes * [0], 0.8,
                                  {'bounds': [-1, 1],
                                   'popsize': POP_SIZE,
                                   'maxiter': N_GENERATIONS})

    while not es.stop():
        print("Generation", es.countiter)
        X = es.ask()  # get list of new solutions
        fit = [simulation(env, x) for x in X]  # evaluate each solution
        es.tell(X, fit)  # besides for termination only the ranking in fit is used
        # es.disp()
    print('termination:', es.stop())
    cma.s.pprint(es.best.__dict__)

    best_solution = es.best.x
    print("Results")
    print("Results pretty")
    print(es.result_pretty())
    print("\n\n---- BEST ----")
    # print(best_solution)
    print(f"Inverted best fitness: {es.best.f}")
    print(f"Inverted best fitness: {simulation(env, best_solution)}")
    print(f"Original best fitness: {simulation(env, best_solution, pure_fitness=True)}")

    # save best solution
    np.savetxt(f'{experiment_name}/{solution_file_name}', best_solution)

    verify_solution(env, best_solution)


if __name__ == "__main__":
    env = Environment(experiment_name=experiment_name,
                      enemies=ENEMIES,
                      multiplemode="yes",
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
