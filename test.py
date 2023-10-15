import numpy as np

solutions = np.loadtxt("C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\BEST_SOLUTION")

shape = solutions.shape
print(shape)
print(solutions.reshape((int(shape[0] / 265), 265)))