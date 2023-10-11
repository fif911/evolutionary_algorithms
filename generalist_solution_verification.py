#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                                 		  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        general solution for enemies (games)                                         #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

import os

# imports other libs
import numpy as np

from demo_controller import player_controller
from evoman.environment import Environment

experiment_name = 'controller_generalist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2,
                  logs="off",
                  visuals=True)

sol = np.loadtxt('solutions_beats_5_enemies/beats_8_enemies_2.txt')

# tests saved demo solutions for each enemy
won_all = True
player_remaining_life_sum = 0
total_time_sum = 0
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])
    f, p, e, t = env.play(sol)
    enemy_beaten = e == 0 and p > 0
    print(f" ----- Enemy {en} Player {p}; Enemy {e}; in {t} seconds. Won: {enemy_beaten}")
    # print(f"Fitness: {f}; Inverted fitness: {1 / f}")
    won_all = won_all and enemy_beaten
    player_remaining_life_sum += p
    total_time_sum += t

print(f"Won all: {won_all}")
print(f"Sum of remaining player life: {player_remaining_life_sum:.2f}/800 (to be maximised)")
print(f"Time took total: {total_time_sum} (to be minimised)")
