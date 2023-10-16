"""
Implementation of multi-objective optimisation using pymoo

https://pymoo.org/algorithms/list.html


Algorithm: SMS-EMOA
Algorithm paper: https://sci-hub.se/https://doi.org/10.1016/j.ejor.2006.08.008
Docs link: https://pymoo.org/algorithms/moo/sms.html

Algorithm: AGE-MOEA
Algorithm paper: https://sci-hub.se/10.1145/3321707.3321839
Docs link: https://pymoo.org/algorithms/moo/age.html#nb-agemoea

"""
import time
import uuid
from typing import Optional

import numpy as np
import pandas as pd
from evoman.environment import Environment

import pymoo.gradient.toolbox as anp
from fitness_functions import just_fight_fitness
from nn_crossover import NNCrossover
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.config import Config
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.visualization.scatter import Scatter
from sketches.diversity_measures import get_most_unique_solutions
from utils import simulation, verify_solution, init_env, initialise_script, print_progress_bar, read_solutions_from_file

Config.warnings['not_compiled'] = False
SMART_INIT = False
if SMART_INIT:
    solutions = read_solutions_from_file("farmed_beats_8")
    next_population = get_most_unique_solutions(population=solutions, n_solutions=10, must_include_ids=(39, 185))
    POP_SIZE = len(next_population)
else:
    next_population = FloatRandomSampling()
    POP_SIZE = 100

ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]

# TERMINATION CRITERIA
term_critea = "n_eval"  # "n_gen" or "n_eval"
if term_critea == "n_gen":
    N_GENERATIONS = 10
elif term_critea == "n_eval":
    N_EVALUATIONS = 50_000
else:
    raise Exception("Invalid termination criteria")

n_hidden_neurons = 10

experiment_name = 'pymoo_sms_emoa'

initialise_script(experiment_name=experiment_name, clean_folder=False)


class objectives(Problem):
    enemies: list[int]
    env: Environment
    last_iteration_evaluation_results: Optional[dict] = None
    last_iteration_objectives_fitness: Optional[dict] = None

    def __init__(self, env: Environment, n_genes: int, enemies: list[int], n_objectives):
        self.env = env
        self.enemies = enemies
        super().__init__(n_var=n_genes, n_obj=n_objectives, xl=-1, xu=1, type_var=float)

    def _evaluate(self, x: list[np.array], out, *args, **kwargs):
        """Evaluate the fitness of each individual in the population
        We can turn on elementwise_evaluation to evaluate each individual in the population separately
        https://pymoo.org/problems/parallelization.html#Custom-Parallelization

        x - list of individuals in the population
        out - dictionary with the fitness outputs

        # when we have multiple enemies we need to average the fitness somehow
        # Ideas: mean of the fitness will show how agent performs with both enemies
        #        max will show how agent performs with the worst enemy (this does not reflect the performance with
        #        another enemy)
        #        weighted average is another option, but then we have another problem of how to weight the enemies
        """

        # Initialize
        dict_enemies = {}
        # Get fitness for each enemy
        for enemy in self.enemies:
            self.env.update_parameter('enemies', [enemy])

            dict_enemies[enemy] = []
            for individual_id in range(len(x)):
                dict_enemies[enemy].append(simulation(self.env, x[individual_id], inverted_fitness=True,
                                                      fitness_function=just_fight_fitness))
        # Return fitness outputs for enemies
        objectives_fitness = {
            "objective_hard": [np.mean([dict_enemies[enemy_id][ind_id] for enemy_id in [1, 6]]) for ind_id in
                               range(len(x))],
            "objective_medium": [np.mean([dict_enemies[enemy_id][ind_id] for enemy_id in [3, 4, 7]]) for ind_id in
                                 range(len(x))],
            "objective_easy": [np.mean([dict_enemies[enemy_id][ind_id] for enemy_id in [2, 5, 8]]) for ind_id in
                               range(len(x))],
        }
        self.last_iteration_evaluation_results = dict_enemies
        self.last_iteration_objectives_fitness = objectives_fitness
        # each enemy is a separate objective
        # objectives_fitness = {
        #     f"objective_{enemy}": dict_enemies[enemy] for enemy in self.enemies
        # }

        out["F"] = anp.column_stack([objectives_fitness[key] for key in objectives_fitness.keys()])


def plot_pareto_fronts(res, best_solutions_idx: list[int], save_plots_to_file=True):
    """Plot the pareto fronts for each pair of objectives and all 3 objectives"""
    print(f"Plotting {res.F.shape[0]} solutions")
    res.F = 1 / res.F

    plot_3d = Scatter(labels=["Hard enemies", "Medium Enemies", "Easy enemies"], title="Pareto Front")
    plot_3d.add(res.F, color="red")
    plot_3d.add(res.F[best_solutions_idx], color="blue", s=80, label="Best solutions")

    # for 3 objectives plot each pair of pareto fronts
    # Hard vs Medium
    plot_hard_medium = Scatter(labels=["Hard enemies", "Medium Enemies"], title="Pareto Front")
    plot_hard_medium.add(res.F[:, [0, 1]], color="red")
    plot_hard_medium.add(res.F[:, [0, 1]][best_solutions_idx], color="blue", s=80, label="Best solutions")

    # Hard vs Easy
    plot_hard_easy = Scatter(labels=["Hard enemies", "Easy Enemies"], title="Pareto Front")
    plot_hard_easy.add(res.F[:, [0, 2]], color="red")
    plot_hard_easy.add(res.F[:, [0, 2]][best_solutions_idx], color="blue", s=80, label="Best solutions")

    # Medium vs Easy
    plot_medium_easy = Scatter(labels=["Medium enemies", "Easy Enemies"], title="Pareto Front")
    plot_medium_easy.add(res.F[:, [1, 2]], color="red")
    plot_medium_easy.add(res.F[:, [1, 2]][best_solutions_idx], color="blue", s=80, label="Best solutions")

    file_names_and_plots = [("plot_3d", plot_3d), ("plot_hard_medium", plot_hard_medium),
                            ("plot_hard_easy", plot_hard_easy),
                            ("plot_medium_easy", plot_medium_easy)]
    run_uuid = uuid.uuid4()
    for plot_name, plot in file_names_and_plots:
        if save_plots_to_file:
            plot.save(f"{experiment_name}/pareto_fronts_{plot_name}_{run_uuid}.png", dpi=300)
        else:
            plot.show()


def main(env: Environment, n_genes: int, population=None):
    start_time = time.time()
    problem = objectives(
        env=env,
        n_genes=n_genes,
        enemies=ENEMIES,
        n_objectives=3
    )

    print("Starting algorithm...; Population size: ", POP_SIZE)

    algorithm = SMSEMOA(pop_size=POP_SIZE, sampling=population, crossover=NNCrossover())
    if term_critea == "n_eval":
        algorithm.setup(problem, termination=('n_eval', N_EVALUATIONS), verbose=False)
    elif term_critea == "n_gen":
        algorithm.setup(problem, termination=('n_gen', N_GENERATIONS), verbose=False)

    datastore = []  # list with: (n_gens, n_evals, ind_id, f1, f2, f3, f4, f5, f6, f7, f8, obj_hard, obj_medium, obj_easy)
    while algorithm.has_next():
        # ask the algorithm for the next solution to be evaluated
        pop = algorithm.ask()
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        algorithm.evaluator.eval(problem, pop)
        # save the fitness values to a datastore
        for ind_id, _ in enumerate(pop):
            ind_fitness = [1 / problem.last_iteration_evaluation_results[enemy][ind_id] for enemy in
                           problem.last_iteration_evaluation_results.keys()]
            ind_objective_fitness = [1 / problem.last_iteration_objectives_fitness[objective][ind_id] for objective in
                                     problem.last_iteration_objectives_fitness.keys()]
            datastore_row = [algorithm.n_gen, algorithm.evaluator.n_eval, ind_id] + ind_fitness + ind_objective_fitness
            datastore.append(datastore_row)

        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)
        if term_critea == "n_eval":
            print_progress_bar(algorithm.evaluator.n_eval, total=N_EVALUATIONS, start_time=start_time)
        elif term_critea == "n_gen":
            print_progress_bar(algorithm.n_gen, total=N_GENERATIONS, start_time=start_time)

    print()
    # obtain the result objective from the algorithm
    res = algorithm.result()
    print(f"Algorithm result contains {len(res.X)} solutions")

    max_enemies_beaten = 0
    best_solutions = []
    best_solutions_idx = []
    for i, x in enumerate(res.X):
        enemies_beaten = verify_solution(env, x, enemies=[1, 2, 3, 4, 5, 6, 7, 8], print_results=False)
        if enemies_beaten > max_enemies_beaten:
            max_enemies_beaten = enemies_beaten
            best_solutions = [x]  # reset the list because we found a better performing solution
            best_solutions_idx = [i]
        elif enemies_beaten == max_enemies_beaten:
            best_solutions.append(x)  # add to the list the solution that beats the same number of enemies
            best_solutions_idx.append(i)

    print(f"Most enemies beaten: {max_enemies_beaten}; Number of these solutions: {len(best_solutions)}")
    print(f"Individuals evaluated: {algorithm.evaluator.n_eval}")

    if max_enemies_beaten == 8:
        # save the best solutions to files only if they beat all enemies
        for i, solution in enumerate(best_solutions):
            np.savetxt(f'{experiment_name}/enemies_beaten_{max_enemies_beaten}_{i}_{uuid.uuid4()}.txt', solution)

    plot_pareto_fronts(res, best_solutions_idx)
    return datastore


if __name__ == '__main__':
    print("Running pymoo_sms_emoa.py")
    env, n_genes = init_env(experiment_name, ENEMIES, n_hidden_neurons)
    env.update_parameter('multiplemode', 'no')

    datastore = main(env, n_genes, population=next_population)
    df = pd.DataFrame(datastore, columns=[
        'n_gens', 'n_evals', 'ind_id', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8',
        'obj_hard', 'obj_medium', 'obj_easy'
    ])
    df.to_csv(f"{experiment_name}/pymoo_sms_emoa_datastore_{uuid.uuid4()}.csv", index=False)

    print("Done!")
