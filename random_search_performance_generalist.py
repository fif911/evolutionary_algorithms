import time
import uuid

import pandas as pd

from utils import initialise_script, init_env, verify_solution, print_progress_bar, calculate_ind_score

experiment_name = 'random_search_performance_generalist'
initialise_script(experiment_name, clean_folder=False)
import numpy as np

ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]
if __name__ == '__main__':
    start_time = time.time()
    env, n_genes = init_env(experiment_name, ENEMIES, 10)
    POP_SIZE = 50_000
    random_solutions = np.random.uniform(-1, 1, (POP_SIZE, n_genes))
    print(f"Population size: {POP_SIZE}")

    population = np.random.uniform(-1, 1, (POP_SIZE, n_genes))

    # pd array with columns: score, enemies beaten
    scores = []
    for idx, x in enumerate(random_solutions):
        enemies_beaten, enemies_not_beaten, enemy_lives, player_lives, times = verify_solution(env, x,
                                                                                               enemies=ENEMIES,
                                                                                               vv=True,
                                                                                               print_results=False)
        # compose aggregate fitness value
        score = calculate_ind_score(enemies_beaten=enemies_beaten, enemy_lives=enemy_lives, enemies=ENEMIES)
        scores.append(
            (score, len(enemies_beaten), enemies_beaten, enemies_not_beaten, enemy_lives, player_lives, times)
        )
        print_progress_bar(idx + 1, total=POP_SIZE, start_time=start_time)

    # convert to pd array and save to csv. To the filename add pop size and uuid in the end.
    df = pd.DataFrame(scores, columns=['score', 'enemies beaten', 'enemies beaten list', 'enemies not beaten list',
                                       'enemy lives', 'player lives', 'times'])
    df.to_csv(f"{experiment_name}/random_search_performance_generalist_{POP_SIZE}_{uuid.uuid4()}.csv")
    print("Done.")
