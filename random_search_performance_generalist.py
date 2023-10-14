import time
import uuid

import pandas as pd

from utils import initialise_script, init_env, verify_solution

experiment_name = 'random_search_performance_generalist'
initialise_script(experiment_name, clean_folder=False)
import numpy as np


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


ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]
if __name__ == '__main__':
    start_time = time.time()
    env, n_genes = init_env(experiment_name, ENEMIES, 10)
    POP_SIZE = 100_000
    random_solutions = np.random.uniform(-1, 1, (POP_SIZE, n_genes))
    print(f"Population size: {POP_SIZE}")

    population = np.random.uniform(-1, 1, (POP_SIZE, n_genes))

    # pd array with columns: score, enemies beaten
    scores = []
    for idx, x in enumerate(random_solutions):
        enemies_beaten, enemies_not_beaten, enemy_lives, player_lives, times = verify_solution(env, x,
                                                                                               enemies=[1, 2, 3, 4, 5,
                                                                                                        6, 7, 8],
                                                                                               vv=True,
                                                                                               print_results=False)
        # compose aggregate fitness value
        score = len(enemies_beaten)
        for i_enemy in range(len(ENEMIES)):
            # For example if enemy live and all enemies to beat were 3
            # enemy life: 80 --> (100 - 80) / 100 * 3 = 20/300 --> 0.6
            # enemy life: 20 --> (100 - 20) / 100 * 3 = 80/300 --> 2.4
            score += (100 - enemy_lives[i_enemy]) / (100 * len(ENEMIES))  # Also count evaluated enemies
        scores.append(
            (score, len(enemies_beaten), enemies_beaten, enemies_not_beaten, enemy_lives, player_lives, times)
        )
        print_progress_bar(idx + 1, total=POP_SIZE, start_time=start_time)

    # convert to pd array and save to csv. To the filename add pop size and uuid in the end.
    df = pd.DataFrame(scores, columns=['score', 'enemies beaten', 'enemies beaten list', 'enemies not beaten list',
                                       'enemy lives', 'player lives', 'times'])
    df.to_csv(f"{experiment_name}/random_search_performance_generalist_{POP_SIZE}_{uuid.uuid4()}.csv")
    print("Done.")
