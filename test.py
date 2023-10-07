from demo_controller import player_controller
from evoman.environment import Environment
import numpy as np
import os
import pymoo

from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.operators.sampling.rnd import random_by_bounds

from pymoo.core.problem import Problem
import pymoo.gradient.toolbox as anp
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


N_GENERATIONS = 50
POP_SIZE = 100
ENEMIES = [6, 2, 5, 7, 8]
MODE = "train"  # train or test

n_hidden_neurons = 10

experiment_name = 'cma_v2_test'
solution_file_name = 'cma_v2_best.txt'
os.environ["SDL_VIDEODRIVER"] = "dummy"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                    enemies=ENEMIES,
                    multiplemode= "no",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    logs="off",
                    visuals=False)


def simulation(env, xm: np.ndarray, pure_fitness=False, return_enemies=False):
    """Run one episode and return the fitness

    pure_fitness: if True, return the fitness as is, otherwise return the inverse of the fitness for minimization problem
    return_enemies: if True, return the player life, enemy life and time
    """
    f, p, e, t = env.play(pcont=xm)
    if pure_fitness:
        return f
    if return_enemies:
        return p, e, t

    fitness = 0.9 * (100 - e) + 0.1 * p - np.log(t)

    if fitness <= 0:
        fitness = 0.00001

    return 1 / fitness

class objectives(Problem):
    
        def __init__(self):
            super().__init__(n_var=265, n_obj=2, n_constr=0, xl=-1, xu=1, type_var=float)
    
        def _evaluate(self, x, out, *args, **kwargs):
            # Initialize
            dict_enemies = {}
            # Get fitness for each enemy
            for enemy in ENEMIES:
                env.update_parameter('enemies', [enemy])
                dict_enemies[enemy] = []
                for i in range(POP_SIZE):
                    dict_enemies[enemy].append(simulation(env, x[i,:]))
            # Stack fitness
            dict_enemies[2578] = []#, dict_enemies[4] = [], []
            for i in range(POP_SIZE):
                dict_enemies[2578].append(max([dict_enemies[j][i] for j in [2, 5, 7, 8]])) # Temporarily
                #dict_enemies[4].append(max([dict_enemies[j][i] for j in [4]])) # Temporarily

            
            out["F"] = anp.column_stack([dict_enemies[6], dict_enemies[2578]])


problem = objectives()
# from pymoo.problems import get_problem
# problem = get_problem("zdt5")


# algorithm = AGEMOEA(pop_size = POP_SIZE,
#                     sampling=random_by_bounds(n_var = 265, xl = -1, xu = 1, n_samples=POP_SIZE),
#                     crossover=TwoPointCrossover(),
#                     mutation=GaussianMutation(),
#                     eliminate_duplicates=True)
algorithm = SMSEMOA(pop_size = POP_SIZE)

# prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), seed=1, verbose=False)

# until the algorithm has no terminated
while algorithm.has_next():
    
    # ask the algorithm for the next solution to be evaluated
    pop = algorithm.ask()

    # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
    algorithm.evaluator.eval(problem, pop)

    # returned the evaluated individuals which have been evaluated or even modified
    algorithm.tell(infills=pop)

    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(algorithm.n_gen, algorithm.evaluator.n_eval)
    print(1 / algorithm.result().F)

# obtain the result objective from the algorithm
res = algorithm.result()
# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())

res.F = 1 / res.F
print(res.F)

for i, x in enumerate(res.X):
    print("***************************")
    print("Point: ", i)
    for enemy in np.arange(1, 9):
        print("Fighting enemy: ", enemy)
        env.update_parameter('enemies', [enemy])
        p, e, t = simulation(env, x, return_enemies = True)
        if p > 0:
            print("\tWon")
        else:
            print("\tLost")
        

Scatter().add(res.F, facecolor="none", edgecolor="red").show()