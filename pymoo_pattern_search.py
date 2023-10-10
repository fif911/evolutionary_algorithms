from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.problems.single import Himmelblau
from pymoo.optimize import minimize

problem = Himmelblau()

algorithm = PatternSearch()

res = minimize(problem,
               algorithm,
               verbose=False,
               seed=1)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
