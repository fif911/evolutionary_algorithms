import os
from typing import Optional

import numpy as np
from evoman.environment import Environment

from demo_controller import player_controller


def simulation(env, xm: np.ndarray, inverted_fitness=True, verbose=False):
    """Run one episode and return the inverted fitness for minimization problem

    Fitness function:
    fitness = 0.9 * (100 - e) + 0.1 * p - np.log(t)

    pure_fitness: if True, return the fitness as is, otherwise return the inverse of the fitness for minimization problem
    return_enemies: if True, return the player life, enemy life and time
    """
    f, p, e, t = env.play(pcont=xm)
    if not inverted_fitness:
        return f  # return the original fitness
    if verbose:
        return p, e, t

    if f <= 0:
        f = 0.00001

    return 1 / f


def verify_solution(env, best_solution, enemies: Optional[list[int]] = None) -> int:
    """Verify the solution on the given enemies. If enemies is None, then verify on all enemies"""

    if enemies is None:
        enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    enemies_beaten = 0
    env.update_parameter("multiplemode", "no")

    for enemy_idx in enemies:
        env.update_parameter('enemies', [enemy_idx])
        p, e, t = simulation(env, best_solution, verbose=True)
        enemy_beaten = e == 0 and p > 0
        print(
            f"Enemy {enemy_idx};\tPlayer Life: {p:.2f},\t Enemy Life: {e:.2f},\t in {t:.2f} seconds. "
            f"\tWon: {enemy_beaten}")
        if enemy_beaten:
            enemies_beaten += 1
    print(f"Enemies beaten: {enemies_beaten}/{len(enemies)}")

    return enemies_beaten


def init_env(experiment_name, enemies, n_hidden_neurons) -> (Environment, int):
    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,
                      multiplemode="yes" if len(enemies) > 1 else "no",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off",
                      visuals=False)
    n_genes = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    return env, n_genes


def run_pymoo_algorithm(algorithm, problem, experiment_name="pymoo_sms_emoa", postfix=""):
    # until the algorithm has no terminated
    while algorithm.has_next():
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        algorithm.evaluator.eval(problem, pop)
        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)
        # do same more things, printing, logging, storing or even modifying the algorithm object
        print(f"Generation: {algorithm.n_gen}")
        print(f"Best individual fitness: {', '.join([f'{_:.3f}' for _ in algorithm.result().F[0]])}")

    # save the whole population to a file
    np.savetxt(f'{experiment_name}/algorithm_gens_{algorithm.n_gen}_p-size_{len(algorithm.pop)}_{postfix}.txt',
               algorithm.pop.get("X"))

    return algorithm


def initialise_script(experiment_name):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Clean the folder
    for file in os.listdir(experiment_name):
        os.remove(os.path.join(experiment_name, file))

    os.environ["SDL_VIDEODRIVER"] = "dummy"
