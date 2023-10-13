import os

# imports other libs
import numpy as np

from demo_controller import player_controller
from evoman.environment import Environment

from utils import verify_solution, init_env, read_solutions_from_file, initialise_script

experiment_name = 'generalist_solution_verification_multi'

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
# env = Environment(experiment_name=experiment_name,
#                   playermode="ai",
#                   player_controller=player_controller(n_hidden_neurons),
#                   speed="fastest",
#                   enemymode="static",
#                   level=2,
#                   logs="off",
#                   visuals=True)
initialise_script(experiment_name)
env, _ = init_env(experiment_name, enemies=[1, 2, 3, 4, 5, 6, 7, 8], n_hidden_neurons=n_hidden_neurons)

solutions = read_solutions_from_file("magic_8", startswith="beats_8_")
solutions_existing = read_solutions_from_file("farmed_beats_8", startswith="beats_8")
solutions = np.concatenate((solutions, solutions_existing))
print(f"Number of solutions: {len(solutions)}")

max_health = 0
its_time = 0
win_id = None
for id, sol in enumerate(solutions):
    print(f"Solution {id}/{len(solutions)}")
    enemies_beaten, player_lifes, times = verify_solution(env, sol, enemies=[1, 2, 3, 4, 5, 6, 7, 8],
                                                          print_results=False,
                                                          verbose_for_gain=True)
    p_health = sum(player_lifes)
    if p_health > max_health:
        max_health = p_health
        its_time = sum(times)
        win_id = id

    # print(f"Won all: {len(enemies_beaten) == 8}")
    print(f"Sum of remaining player life: {sum(player_lifes):.2f}/800 (to be maximised)")
    # print(f"Time took total: {sum(times)} (to be minimised)")

print("---")
print(f"Max health: {max_health}")
print(f"In Time: {its_time}")
print(f"Win id: {win_id}")

verify_solution(env, solutions[win_id], enemies=[1, 2, 3, 4, 5, 6, 7, 8])
# np.savetxt(f'{experiment_name}/submission_solution.txt', solutions[win_id])
