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
                  speed="normal",
                  enemymode="static",
                  level=2,
                  logs="off",
                  visuals=True)

sol = np.loadtxt('solutions_demo/cma_v2_best_ngens_200_pop_size100.txt')

# tests saved demo solutions for each enemy
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])
    f, p, e, t = env.play(sol)
    print(f" ----- Enemy {en} Player {p}; Enemy {e}; in {t} seconds. Won: {e == 0}")
