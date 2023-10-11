import os

import numpy as np
import scipy


def calc_similarity_matrix(population: np.array, threshold: float = 1, metric: str = "euclidean"):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    The distance metric to use. The distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.

    for euclidial distance, 11 means they are diverse
    lower than 0.5 means they are exactly similar
    5 means they are similar but not exact (50% similar)

    We choose lower than 1 as a threshold

    returns similarity matrix with True/False values
    """
    condensed_matrix = scipy.spatial.distance.pdist(X=population, metric=metric)
    square_matrix = scipy.spatial.distance.squareform(X=condensed_matrix)
    return square_matrix < threshold  # return similarity matrix with True/False values


def number_of_similar_solutions_per_individual(population, threshold, metric="euclidean"):
    """Calculate the number of similar solutions per individual that are lower than the threshold
    for euclidial distance, 11 means they are diverse
    lower than 0.5 means they are exactly similar
    5 means they are similar but not exact (50% similar)


    The resulting array will have the same length as the population and will show how many similar solutions
    are there for each individual.
    One is the minimum (the individual itself) and the maximum is the length of the
    population - 1 (all other individuals are similar to the current one)
    """
    similarity_matrix = calc_similarity_matrix(population, threshold, metric)
    return np.sum(similarity_matrix, axis=1)


def _get_population():
    # Get np array of solutions for testing
    solutions = []
    for file in os.listdir('../solutions_beats_5_enemies'):
        if file.startswith('pymoo'):
            new_solution = []
            with open(f'../solutions_beats_5_enemies/{file}') as f:
                solutions.append(f.read().splitlines())

    solutions = np.array(solutions, dtype=float)
    return solutions


if __name__ == '__main__':
    # test different populations
    population = _get_population()
    population = np.random.uniform(low=-1, high=1, size=(100, 256))  # average euclidean distance is 13
    population = np.random.normal(loc=0, scale=0.1, size=(100, 256))  # average euclidean distance is 2.2
    population = np.random.normal(loc=0, scale=0.05, size=(100, 256))  # average euclidean distance is 1.1

    similarity_matrix = calc_similarity_matrix(population, threshold=1, metric="euclidean")
    per_individual = number_of_similar_solutions_per_individual(population, threshold=1, metric="euclidean")

    print(per_individual)
