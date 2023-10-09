"""
Implementation of multi-objective optimisation using pymoo
"""
import time

import numpy as np
from dask.distributed import Client

from utils import simulation, init_env

if __name__ == '__main__':
    time_start = time.time()

    client = Client(serializers=['dask', 'pickle'],
                    deserializers=['dask', 'msgpack'])
    client.restart()
    print("--- DASK STARTED ---")

    env, n_genes = init_env("", [1], 10)
    random_solutions = np.random.uniform(-1, 1, (50, n_genes))

    # run simulation for each solution in parallel
    results = client.map(simulation, [env] * len(random_solutions), random_solutions)
    results = client.gather(results)
    print(results)

    client.close()
    print(f"Total time (minutes): {(time.time() - time_start) / 60:.2f}")
    print("Done!")
