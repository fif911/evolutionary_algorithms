import numpy as np


def original_fitness(player_life, enemy_life, time):
    """Calculate the original fitness"""
    return 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time)


def individual_gain(player_life, enemy_life, time):
    """Individual gain from 0 to 200 with small penalty for time

    50 - means the player lost
    250 - means the player won without losing any life
    """
    return 250 + (player_life - enemy_life) - np.log(time)
