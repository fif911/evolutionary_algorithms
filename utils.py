from typing import Optional

import numpy as np


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


def verify_solution(env, best_solution, enemies: Optional[list[int]] = None):
    if enemies is None:
        enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    enemies_beaten = 0
    env.update_parameter("multiplemode", "no")

    for enemy in enemies:
        env.update_parameter('enemies', [enemy])
        p, e, t = simulation(env, best_solution, verbose=True)
        enemy_beaten = e == 0 and p > 0
        print(f"Enemy {enemy};\tPlayer Life: {p}, Enemy Life: {e}, in {t} seconds. \tWon: {enemy_beaten}")
        if enemy_beaten:
            enemies_beaten += 1
    print(f"Enemies beaten: {enemies_beaten}/{len(enemies)}")
