import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.variable import Real, get
from pymoo.util.misc import row_at_least_once_true


def mut_binomial(n, m, prob, at_least_once=True):
    prob = np.ones(n) * prob
    M = np.random.random((n, m)) < prob[:, None]

    if at_least_once:
        M = row_at_least_once_true(M)
    return M
    # # Setting
    # option = "hidden in, output in"

    # # Hidden neurons and inputs
    # n_hidden = 10
    # n_inputs = 20
    # n_vars = 265

    # # Mating Pool Initialization
    # mating_pool = np.zeros((n, n_vars))

    # # Binary Hidden Neurons
    # bin_hiddens = np.random.randint(0, 2, (n, n_hidden))
    # for member in range(n): # Always at least one hidden neuron 1 for each member --> change proportion?
    #     if np.sum(bin_hiddens[member, :]) == 0:
    #         bin_hiddens[member, np.random.randint(0, n_hidden)] = 1

    # for member in range(n):
    #     for i in range(n_hidden):
    #         if bin_hiddens[member, i] == 1:
    #             # Bias of the hidden neuron
    #             mating_pool[member, i] = 1
    #             # Incoming weights of the hidden neuron
    #             idx = np.arange(i + n_hidden, n_inputs * n_hidden + n_hidden, n_hidden)
    #             mating_pool[member, idx] = 1
    #             if option == "hidden in and out":
    #                 # Outgoing weights of the hidden neuron
    #                 idx2 = np.arange(n_inputs * n_hidden + n_hidden + 5 + (i * 5), n_inputs * n_hidden + n_hidden + 5 + ((i + 1) * 5), 1)
    #                 mating_pool[member, idx2] = 1

    #     if option == "hidden in and out":
    #         # Final bias --> randomly one or zero
    #         mating_pool[member, n_inputs * n_hidden + n_hidden:n_inputs * n_hidden + n_hidden + 5] = np.random.randint(0, 2, 5)

    #     # Output neurons
    #     if option == "hidden in, output in":
    #         bin_output = np.random.randint(0, 2, (n, 5))
    #         for member in range(n): # Always at least one output 1 for each member --> change proportion?
    #             if np.sum(bin_output[member, :]) == 0:
    #                 bin_output[member, np.random.randint(0, 5)] = 1

    #         for member in range(n):
    #             for i in range(5):
    #                 if bin_output[member, i] == 1:
    #                     # Final bias
    #                     mating_pool[member, n_inputs * n_hidden + n_hidden + i] = 1
    #                     # Incoming weights of the output neuron
    #                     idx2 = np.arange(i + n_inputs * n_hidden + n_hidden + 5, n_vars, 5)
    #                     mating_pool[member, idx2] = 1

    # return mating_pool.astype(int)


class BinomialCrossover(Crossover):

    def __init__(self, bias=0.5, n_offsprings=2, **kwargs):
        super().__init__(2, n_offsprings, **kwargs)
        self.bias = Real(bias, bounds=(0.1, 0.9), strict=(0.0, 1.0))

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        bias = get(self.bias, size=n_matings)
        M = mut_binomial(n_matings, n_var, bias, at_least_once=True)

        if self.n_offsprings == 1:
            Xp = X[0].copy(X)
            Xp[~M] = X[1][~M]
        elif self.n_offsprings == 2:
            Xp = np.copy(X)
            Xp[0][~M] = X[1][~M]
            Xp[1][~M] = X[0][~M]
        else:
            raise Exception
        return Xp


class BX(BinomialCrossover):
    pass
