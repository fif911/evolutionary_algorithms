import numpy as np
import pandas as pd

from utils import read_solutions_from_file, verify_solution, initialise_script, init_env

box_plot_report = "dynamic_7_enemies"

initialise_script(box_plot_report, clean_folder=False)
solutions = read_solutions_from_file(box_plot_report)
env, _ = init_env("test", [1], 10)

# datastore in format [sol_id, enemies_beaten_c, ind_gain]
datastore = []

for idx, x in enumerate(solutions):
    enemies_beaten, _, enemy_lives, player_lifes, _ = verify_solution(env, x, vv=True)
    datastore.append([idx, len(enemies_beaten), sum(np.array(player_lifes) - np.array(enemy_lives))])

# store pd dataframe as csv

df = pd.DataFrame(datastore, columns=["sol_id", "enemies_beaten", "ind_gain"])
df.to_csv(f"{box_plot_report}.csv", index=False)
