import numpy as np
from evoman.environment import Environment
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.problems.single import Himmelblau

from utils import init_env, simulation, initialise_script

POP_SIZE = 10
N_GENERATIONS = 4
ENEMIES = [1]
experiment_name = 'pymoo_pattern_search'
initialise_script(experiment_name=experiment_name)


class objectives(Problem):
    enemies: list[int]
    env: Environment

    def __init__(self, env: Environment, n_genes: int, enemies: list[int]):
        self.env = env
        self.enemies = enemies
        super().__init__(n_var=n_genes, n_obj=1, xl=-1, xu=1, type_var=float)

    def _evaluate(self, x: list[np.array], out, *args, **kwargs):
        fitness = []
        # print("Evaluating")
        for individual_id in range(len(x)):
            fitness.append(simulation(self.env, x[individual_id], inverted_fitness=True))

        out["F"] = fitness


if __name__ == '__main__':
    env, n_genes = init_env(experiment_name, ENEMIES, 10)

    algorithm = PatternSearch(verbose=True)

    # problem = objectives(env, n_genes, ENEMIES)
    problem = Himmelblau()
    current_pop = np.random.uniform(-1, 1, (POP_SIZE, n_genes))

    algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), n_sample_points=POP_SIZE, )

    minimize(problem, algorithm, ('n_gen', N_GENERATIONS), verbose=True, seed=1)
    while algorithm.has_next():
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()
        # TODO: Problem with the population size here
        print(f"LEN POP AFTER ASK {len(pop)}")

        assert len(pop) == POP_SIZE

        # print(len(pop))
        current_pop = pop
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        algorithm.evaluator.eval(problem, pop)
        # returned the evaluated individuals which have been evaluated or even modified
        print(f"LEN POP BEFORE TELL {len(pop)}")
        algorithm.tell(infills=pop)
        print(f"LEN POP AFTER TELL {len(pop)}")

    # do same more things, printing, logging, storing or even modifying the algorithm object
    print(f"Generation: {algorithm.n_gen}")
    print(f"Not improved steps: {algorithm.n_not_improved}")
    best_idx = algorithm.pop.get('F').argmin()
    print(f"Best individual fitness: {algorithm.pop.get('F')[best_idx][0]:.3f}")
