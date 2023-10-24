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


def just_fight_fitness(player_life, enemy_life, time):
    """Prioritize hitting the enemy over staying alive

    Does not consider time
    enemy life 0; player_life 20 --> 100 - 0 + 2.99 = 102.99
    enemy life 0; player_life 100 --> 100 - 0 + 4.6 = 104.6
    """
    return (100 - enemy_life) + np.log(player_life + 0.001)


def destroy_and_save_player_life(player_life, enemy_life, time):
    """Prioritize hitting the enemy over staying alive

    Does not consider time
    enemy life 0; player_life 20 --> 100 - 0 + 2.99 = 102.99
    enemy life 0; player_life 100 --> 100 - 0 + 4.6 = 104.6

    max value of np.log(100) is ±5
    """
    return (100 - enemy_life) + player_life / 10
