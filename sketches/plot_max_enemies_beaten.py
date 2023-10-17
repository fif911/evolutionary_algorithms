import pandas as pd
from matplotlib import pyplot as plt

from utils import read_solutions_from_file

series_sets = read_solutions_from_file("../final_generalist_assignment", startswith="max_")
print(series_sets.shape)  # (5, 251)

# Plot the series
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(series_sets.shape[0]):
    ax.plot(series_sets[i], label=f"Experiment {i + 1}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max enemies beaten")
    ax.set_title("Max enemies beaten per iteration dynamic objectives algorithm")
    ax.legend()
plt.show()

# series_sets = pd.read_csv("../pymoo_sms_emoa/pymoo_sms_emoa_datastore0_8e061a4d-fd7d-4376-b1af-58ab8c2cc1ca.csv")
# print(series_sets.shape)
# f = [f'f{i}' for i in range(1, 9)]
# agg = series_sets[f].aggregate(['mean'], axis=1)
# agg_mean = series_sets[['n_gens', 'n_evals', 'ind_id']].join(agg)[['n_evals', 'mean']]
# agg_max = agg_mean.groupby('n_evals').max().rename(columns={'mean': 'max'})
# print()
