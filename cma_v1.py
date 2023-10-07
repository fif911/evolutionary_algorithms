"""
Install cmaes package:
pipenv install cmaes
or
pip install cmaes
"""
import os
from typing import Optional

import numpy as np
from cmaes import CMA

from demo_controller import player_controller
from evoman.environment import Environment

from utils import simulation, verify_solution

n_hidden_neurons = 10

experiment_name = 'cma_test'
os.environ["SDL_VIDEODRIVER"] = "dummy"

ENEMIES = [7]
N_GENERATIONS = 20
POP_SIZE = 50


def norm(f: int, pfit_pop: list[float]):
    """Normalize fitness f based on the population fitness pfit_pop

    Input:
        x: Fitness value, float
        pfit_pop: Population fitness, np.array

    Output:
        x_norm: Normalized fitness, float
    """
    # If not all the same
    if (max(pfit_pop) - min(pfit_pop)) > 0:
        x_norm = (f - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
    else:
        x_norm = 0

    # If negative
    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


def main():
    env = Environment(experiment_name=experiment_name,
                      enemies=ENEMIES,
                      playermode="ai",
                      multiplemode="yes" if len(ENEMIES) > 1 else "no",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off",
                      visuals=False)

    N_GENES = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    bounds = np.concatenate(
        [
            np.tile([-1, 1], (N_GENES, 1)),
        ]
    )

    optimizer = CMA(mean=np.zeros(N_GENES), sigma=0.8, population_size=POP_SIZE, bounds=bounds)

    while True:
        solutions_x, solutions_f = [], []
        for _ in range(optimizer.population_size):
            xm = optimizer.ask()
            fitness = simulation(env, xm, inverted_fitness=False)
            solutions_x.append(xm)
            solutions_f.append(fitness)

        print(
            f"Generation {optimizer.generation}: Best fitness: {max(solutions_f):.4f},\t "
            f"mean: {np.mean(solutions_f):.4f},\t "
            f"worst: {min(solutions_f):.2f},\t std: {np.std(solutions_f):.1f}")

        # Normalize fitness
        solutions_f_norm = [norm(f, solutions_f) for f in solutions_f]
        solutions_f_norm = [1 / x for x in solutions_f_norm]  # Invert fitness for minimization problem
        solutions = [(x, f) for x, f in zip(solutions_x, solutions_f_norm)]

        optimizer.tell(solutions)

        if optimizer.should_stop() or optimizer.generation >= N_GENERATIONS:
            print("Stop")
            # search where the fitness is the lowest and return the solution
            best_solution = solutions[np.argmin([s[1] for s in solutions])][0]
            print("Best solution: \n")
            print(best_solution)
            print("\nFitness:")
            print(f"CMA Inverse Fitness: {simulation(env, best_solution, inverted_fitness=True):.2f}")
            print(f"Original Fitness: {simulation(env, best_solution, inverted_fitness=False):.2f}")
            break

    # Enemies beaten?
    verify_solution(env, best_solution)


if __name__ == "__main__":
    print("Starting...")
    main()
    print("Done!")
