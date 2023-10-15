import os
import time
from typing import Optional, Callable

import numpy as np
from evoman.environment import Environment

from demo_controller import player_controller


def simulation(env: Environment, xm: np.ndarray, inverted_fitness=True, verbose=False,
               fitness_function: Callable = None):
    """Run one episode and return the inverted fitness for minimization problem

    Fitness function:
    fitness = 0.9 * (100 - e) + 0.1 * p - np.log(t)

    pure_fitness: if True, return the fitness as is, otherwise return the inverse of the fitness for minimization problem
    return_enemies: if True, return the player life, enemy life and time
    """
    f, p, e, t = env.play(pcont=xm)
    f = (100 - e) + np.log(p + 0.001)
    if fitness_function:
        f = fitness_function(player_life=p, enemy_life=e, time=t)

    if not inverted_fitness:
        return f  # return the original fitness
    if verbose:
        return p, e, t

    if f <= 0:
        f = 0.00001

    return 1 / f


def verify_solution(env: Environment, best_solution, enemies: Optional[list[int]] = None, print_results=True,
                    verbose=False, verbose_for_gain=False):
    """Verify the solution on the given enemies. If enemies is None, then verify on all enemies"""

    if enemies is None:
        enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    env.update_parameter("multiplemode", "no")
    env.update_parameter("randomini", "no")

    enemies_beaten, enemies_not_beaten, player_lifes, enemy_lives, times = [], [], [], [], []

    for enemy_idx in enemies:
        env.update_parameter('enemies', [enemy_idx])
        p, e, t = simulation(env, best_solution, verbose=True)
        is_enemy_beaten = e == 0 and p > 0
        if print_results:
            print(
                f"Enemy {enemy_idx};\tPlayer Life: {p:.2f},\t Enemy Life: {e:.2f},\t in {t:.2f} seconds. "
                f"\tWon: {is_enemy_beaten}")
        if is_enemy_beaten:
            enemies_beaten.append(enemy_idx)
        else:
            enemies_not_beaten.append(enemy_idx)
        enemy_lives.append(e)
        player_lifes.append(p)
        times.append(t)
    if print_results:
        print(f"Enemies beaten: {enemies_beaten}; {len(enemies_beaten)}/{len(enemies)}")
    if verbose_for_gain:
        return enemies_beaten, player_lifes, times
    if verbose:
        return enemies_beaten, enemies_not_beaten, enemy_lives
    else:
        return len(enemies_beaten)


def init_env(experiment_name, enemies, n_hidden_neurons, random_init_place: bool = False) -> (Environment, int):
    env = Environment(experiment_name=experiment_name,
                      enemies=enemies,
                      multiplemode="yes" if len(enemies) > 1 else "no",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      logs="off",
                      visuals=False,
                      randomini="yes" if random_init_place else "no")
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
        print(f"Generation: {algorithm.n_gen}. Evaluations: {algorithm.evaluator.n_eval}")
        print(f"Best individual fitness: {', '.join([f'{v:.3f} ({1 / v:.2f})' for v in algorithm.result().F[0]])}")

    # save the whole population to a file
    # np.savetxt(f'{experiment_name}/algorithm_gens_{algorithm.n_gen}_p-size_{len(algorithm.pop)}_{postfix}.txt',
    #            algorithm.pop.get("X"))

    return algorithm


def fitness_proportional_selection(population, fitness, n_parents, inverted_fitness=True):
    """Fitness proportional selection (roulette wheel selection)"""
    SMOOTHING_FACTOR = 10
    if inverted_fitness:
        fitness = 1 / fitness
    fitness = fitness + SMOOTHING_FACTOR - np.min(fitness)
    fps = fitness / np.sum(fitness)
    selection_probabilities = fps
    parent_indices = np.random.choice(np.arange(0, population.shape[0]), n_parents, p=selection_probabilities)
    return population[parent_indices]


def tournament_selection(population, fitness, n_parents, k=5):
    """
    Tournament selection for a minimization problem

    :param population: 2D NumPy array of shape (n_individuals, n_genes)
    :param fitness: 1D NumPy array of shape (n_individuals,)
    :param n_parents: Number of parents to select
    :param k: Tournament size (default is 5)
    :return: 2D NumPy array of selected parents of shape (n_parents, n_genes)
    """
    if len(population) == n_parents:
        print("Warning: population size is equal to the number of parents to select. ")
        return population
    if len(population) < n_parents:
        raise ValueError("Population size must be greater than the number of parents to select. "
                         f"pop size: {len(population)}, n_parents: {n_parents}")

    selected_parents = []

    # Create a list of indices to track selected individuals
    selected_indices = []
    print(f"Population shape: {population.shape}")
    print(f"Fitness shape: {fitness.shape}")
    print(f"Number of parents: {n_parents}")

    # Repeat the tournament process until n_parents parents are drawn
    while len(selected_parents) < n_parents:
        # Randomly select k unique individuals from the population
        remaining_indices = list(set(range(population.shape[0])) - set(selected_indices))
        if len(remaining_indices) < k:
            # Not enough remaining individuals to form a complete tournament
            # Draw from the entire population (allow duplicates)
            tournament_indices = np.random.choice(population.shape[0], k, replace=True)
            print("Warning: not enough remaining individuals to form a complete tournament")
        else:
            # There are enough remaining individuals for a complete tournament
            individuals_to_draw = min(n_parents - len(selected_parents), k)
            tournament_indices = np.random.choice(remaining_indices, individuals_to_draw, replace=False)

        selected_indices.extend(tournament_indices)
        tournament_candidates = population[tournament_indices]

        # Find the index of the winner (individual with the lowest fitness)
        winner_index = np.argmin(fitness[tournament_indices])

        # Add the winner to the selected parents
        selected_parents.append(tournament_candidates[winner_index])
    return np.array(selected_parents)


def read_solutions_from_file(filepath, startswith=None):
    solutions = []
    for file in os.listdir(filepath):
        with open(f'{filepath}/{file}') as f:
            if startswith and file.startswith(startswith):
                solutions.append(f.read().splitlines())
            if not startswith:
                solutions.append(f.read().splitlines())

    solutions = np.array(solutions, dtype=float)
    return solutions


def initialise_script(experiment_name, clean_folder=True):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Clean the folder
    if clean_folder:
        for file in os.listdir(experiment_name):
            os.remove(os.path.join(experiment_name, file))

    os.environ["SDL_VIDEODRIVER"] = "dummy"


def print_progress_bar(iteration, total, start_time, bar_length=100):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))

    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60.0
    estimated_total_time = elapsed_time / progress if progress > 0 else 0
    estimated_remaining_time = (estimated_total_time - elapsed_time) / 60.0

    print(f'\r[{arrow}{spaces}] {np.round(progress * 100, 3)}% '
          f'\t\tElapsed: {np.round(elapsed_minutes, 2)} min '
          f'\t\tETA: {np.round(estimated_remaining_time, 2)} min', end='')


def calculate_ind_score(enemies_beaten, enemy_lives, enemies: Optional[list[int]] = None):
    # compose aggregate fitness value
    if not enemies:
        enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    score = len(enemies_beaten)
    for i_enemy in range(len(enemies)):
        # For example if enemy live and all enemies to beat were 3
        # enemy life: 80 --> (100 - 80) / 100 * 3 = 20/300 --> 0.6
        # enemy life: 20 --> (100 - 20) / 100 * 3 = 80/300 --> 2.4
        score += (100 - enemy_lives[i_enemy]) / (100 * len(enemies))  # Also count evaluated enemies
