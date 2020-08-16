import itertools
import time
from CliffordAlgebras import Clifford
import DiracOperator
import numpy as np


# Some dummy parameters for use when testing:
p = 3
q = 1
N = 10
g2 = -3.7
g4 = 1

m = np.random.standard_normal((N, N))
n = np.random.standard_normal((N, N))
M = m + n * 1j
M = M + M.conj().T
np.array_equal(M.conj().T, M)
# Get the clifford algebra for the model we are examining.
cliff = Clifford(p, q)
cliff.introduce()

# Seed the random generator with the time current time, so we always get new random numbers
t_raw = int(time.time() * 1000.0)
t_str = list(str(t_raw))[5:]
t = int("". join(t_str))
np.random.seed(t)
odd_products = cliff.get_odd_gamma_products()
# odd_products = np.array(odd_products, order='C')
odd_products[0]
# %%
# Initialise D: we set the weight A so we start with a much wide range of the parameter space

weightA = 1 / 10  # If we use too large a value here, then the proposed starting Dirac causes the weight action to have a large imaginary component. So start with it ~1

D = DiracOperator.random_Dirac_op(odd_products, N, weightA, cliff.matdim)
# D = D1
# D

"""
# We then set the variable weightA to be very small, so that the steps between Diracs is relatively small. There is a balance between having a small enough step size so that many Diracs are accepted and we "follow the valley downwards" and having a large enough step size so that we can escape local minima.

This value will change for each type, it seems to be a hyperparameter of this setup, i.e. requires tuning.

weightA values for the types that give acceptance_rate/num_moves ~50%

(2,0) -> 1./np.power(cliff.matdim,3)/25
(1,3) -> 1./e-2/6 for size 10x10 - takes 30mins to do 40,000 moves.
"""

weightA = 1e-2 / 6
print(weightA)
num_moves = 0
acceptance_rate = 0
for i in range(2):
    for j in range(1000):
        # Run the algorithm for 1000 steps
        D, acceptance_rate = DiracOperator.update_Dirac(
            D, g2, g4, weightA, acceptance_rate, N, cliff.matdim, odd_products)
        num_moves += 1
    # Every 1000s steps, we investigate a little. We print the accepted rate and the action of the Dirac at this moment.
    print(acceptance_rate, num_moves, acceptance_rate / num_moves)
    print(DiracOperator.action(D, g2, g4))
    # np.linalg.eigvals(D)

"""
It would be helpful to store the last Dirac, so we can "continue" the simulation after it reaches 20,0000 steps. For instance, for the type (1,3), 20,000 steps doesnt seem to be enough. 40,0000 seems to settle in a well.
But to have the thermalisation time be ~1%, then we need to continue the simulation for much longer than that.
"""

# We can then "initialise" the Dirac with the last Dirac operator from the simulation if we want to continnue the simulation.
D1 = D


# %%
"""
Idea playground
"""

# weightA = 1. / 6. / pow(cliff.matdim, 1.5)
# D = random_Dirac_op(odd_products, N, weightA)
# D.H == D
# print(D.shape)
# np.trace(D * D)

# old_D = D
# D = random_Dirac_op(odd_products, N, weightA)
# # for i in range(10):
# #     list_D.append(random_Dirac_op(get_odd_products, N))
# #     print(random_Dirac_op(get_odd_products, N))
# print(random_Dirac_op(odd_products, N, weightA))
# # np.trace(list_D[0]*list_D[0])==np.trace(list_D[1]*list_D[0])
# S_old = action(old_D, g2, g4)
# print(S_old)
# S_new = action(old_D + D, g2, g4)
# print(S_new)
# print(S_new < S_old)
# dS = action(D, g2, g4)
# np.real(dS)
# np.exp(dS)


# %%
