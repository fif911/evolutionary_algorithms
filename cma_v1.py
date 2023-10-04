"""
Install cmaes package:
pipenv install cmaes
or
pip install cmaes
"""
import os

import numpy as np
from cmaes import CMA

from demo_controller import player_controller
from evoman.environment import Environment

n_hidden_neurons = 10

experiment_name = 'cma_test'
os.environ["SDL_VIDEODRIVER"] = "dummy"

N_GENERATIONS = 30
POP_SIZE = 50


def simulation(env, xm: np.ndarray, pure_fitness=False):
    """Run one episode and return the fitness

    pure_fitness: if True, return the fitness as is, otherwise return the inverse of the fitness for minimization problem
    """
    f, p, e, t = env.play(pcont=xm)
    if pure_fitness:
        return f
    fitness = 0.9 * (100 - e) + 0.1 * p - np.log(t)
    if fitness <= 0:
        fitness = 0.00001
    return 1 / fitness


def main():
    env = Environment(experiment_name=experiment_name,
                      enemies=[6],
                      playermode="ai",
                      multiplemode="yes",
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
    print(" g    f(x1,x2)     x1      x2  ")
    print("===  ==========  ======  ======")

    while True:
        solutions = []
        print(f"Generation {optimizer.generation}")
        for _ in range(optimizer.population_size):
            xm = optimizer.ask()
            fitness = simulation(env, xm)
            solutions.append((xm, fitness))
            print(
                f"{optimizer.generation:3d}  {fitness:10.5f}"
                f"  {xm[0]:6.2f}  {xm[1]:6.2f}"
            )
        optimizer.tell(solutions)

        if optimizer.should_stop() or optimizer.generation >= N_GENERATIONS:
            print("Stop")
            # search where the fitness is the highest and return the solution
            best_solution = solutions[np.argmin([s[1] for s in solutions])][0]
            print("Best solution: \n")
            print(best_solution)
            print("\nFitness:")
            print(f"CMA Inverse Fitness: {simulation(env, best_solution, pure_fitness=False):.2f}")
            print(f"Original Fitness: {simulation(env, best_solution, pure_fitness=True):.2f}")
            break


if __name__ == "__main__":
    main()
    print("Done!")
