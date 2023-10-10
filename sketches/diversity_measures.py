import numpy as np
import scipy


def distance_matrix(population: np.array, metric: str):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    The distance metric to use. The distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
    """
    return scipy.spatial.distance.pdist(X=population, metric=metric)


def get_number_of_similar_solutions(population: np.ndarray, threshold: float = 0.1, metric: str = "euclidean"):
    """Return the number of similar solutions in the population"""
    distances = distance_matrix(population, metric=metric)
    return np.sum(distances < threshold)


def get_similarity_matrix(population: np.ndarray, absolute_tolerance: float = 0.1):
    """Return the number of similar solutions in the population using np.allclose
    https://numpy.org/doc/stable/reference/generated/numpy.allclose.html#numpy-allclose
    """
    similarity_matrix = np.zeros((len(population), len(population)))
    for i, x in enumerate(population):
        for j, y in enumerate(population):
            similarity_matrix[i, j] = np.allclose(x, y, atol=absolute_tolerance)

    return similarity_matrix


def genotypic_diversity(population):
    """
    Calculate genotypic diversity for a population of genotypes.

    Parameters:
    - population: A list of genotypes, where each genotype is a list of floats.

    Returns:
    - diversity_score: Genotypic diversity score (Shannon entropy).
    """
    num_individuals = len(population)
    num_genes = len(population[0])

    # Count the frequency of each unique gene across the population.

    for genotype in population:
        for gene in genotype:
            gene_frequencies[gene] = gene_frequencies.get(gene, 0) + 1

    # Calculate Shannon entropy.
    diversity_score = 0.0
    for frequency in gene_frequencies.values():
        probability = frequency / (num_individuals * num_genes)
        diversity_score -= probability * np.log(probability)

    return diversity_score


def index_specific_genotypic_diversity(population):
    """
    Calculate diversity based on the occurrences of gene values at the same index in the population.

    Parameters:
    - population: A list of genotypes, where each genotype is a list of floats.

    Returns:
    - diversity_score: Index-specific genotypic diversity score.
    """
    num_individuals = len(population)
    num_genes = len(population[0])

    # Count the number of unique gene values at each index.
    unique_counts = [len(set(genotype[i] for genotype in population)) for i in range(num_genes)]

    # Calculate the diversity score based on the number of unique values at each index.
    diversity_score = sum(unique_counts) / num_genes

    return diversity_score


# Example usage:
# Suppose you have a population of genotypes (lists of floats).
population = [
    [10, 0.4, 0.6, 1001],
    [10, 0.4, 0.6, 1001],
    [10, 0.4, 0.6, 1001],
    [10, 0.3, 0.5, 100],
    [10, 0.4, 0.6, 0.8],
    [0.5, 0.5, 0.7, 10],
]

# Calculate genotypic diversity.
diversity_score = index_specific_genotypic_diversity(population)

# The diversity_score represents the genotypic diversity of the population.
print(diversity_score)
