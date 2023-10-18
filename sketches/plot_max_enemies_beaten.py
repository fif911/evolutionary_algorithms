import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import read_solutions_from_file

# series_sets = read_solutions_from_file("../final_generalist_assignment", startswith="max_")
# print(series_sets.shape)  # (5, 251)
#
# # Plot the series
# fig, ax = plt.subplots(figsize=(10, 6))
# for i in range(series_sets.shape[0]):
#     ax.plot(series_sets[i], label=f"Experiment {i + 1}")
#     ax.set_xlabel("Iteration")
#     ax.set_ylabel("Max enemies beaten")
#     ax.set_title("Max enemies beaten per iteration dynamic objectives algorithm")
#     ax.legend()
# plt.show()

series_sets = pd.read_csv(
    "../final_generalist_assignment/dynamic_objectives_5_94e231a0-2e4a-4af0-9c1e-dc5e12444655.csv")
print(series_sets.shape)
# f = [f'f{i}' for i in range(1, 9)]
# agg = series_sets[f].aggregate(['mean'], axis=1)
# agg_mean = series_sets[['n_gens', 'n_evals', 'ind_id']].join(agg)[['n_evals', 'mean']]
# agg_max = agg_mean.groupby('n_evals').max().rename(columns={'mean': 'max'})
# print()
series_sets['mean_obj'].to_numpy()
array = np.array(series_sets['mean_obj']).reshape(int(series_sets.shape[0] / 20), 20)
print(array.shape)
fitness_mean = np.mean(array, axis=1)
fitness_max = np.max(array, axis=1)
fitness_std = np.std(array, axis=1)

# plot the fitness
# fig, ax = plt.subplots(figsize=(10, 6))
#
# ax.plot(fitness_mean, label="Mean")
# ax.plot(fitness_max, label="Max")
# ax.fill_between(range(len(fitness_mean)), fitness_mean - fitness_std, fitness_mean + fitness_std, alpha=0.2)
# ax.set_xlabel("Generation")
# ax.set_ylabel("Fitness")
# ax.set_title("Fitness per generation dynamic objectives algorithm")
# ax.legend()
# plt.show()

max_enemies_beaten_over_iterations = np.loadtxt(
    "../final_generalist_assignment/max_enemies_beaten_5_94e231a0-2e4a-4af0-9c1e-dc5e12444655.txt")
max_enemies_beaten_over_iterations = max_enemies_beaten_over_iterations.astype(int)
print(max_enemies_beaten_over_iterations.shape)
# duplicate the entries 20 time to match the number of iterations
max_enemies_beaten_over_iterations = np.repeat(max_enemies_beaten_over_iterations, 9)

# plot the max enemies beaten over iterations on right hand side and fitness on left hand side
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fitness_mean, label="Mean", color="gray")
ax.plot(fitness_max, label="Max", color="black")
ax.fill_between(range(len(fitness_mean)), fitness_mean - fitness_std, fitness_mean + fitness_std, alpha=0.2)
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")
ax.set_title("Fitness per generation dynamic objectives algorithm")
ax.legend()
ax2 = ax.twinx()
ax2.plot(max_enemies_beaten_over_iterations, label="Max enemies beaten", color="orange")
ax2.set_ylabel("Max enemies beaten")
ax2.legend()
plt.show()
