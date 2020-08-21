import itertools
import time
from CliffordAlgebras import Clifford
import DiracOperator
import numpy as np
from numba import njit
# Some dummy parameters for use when testing:
p = 3
q = 1
N = 10
g2 = -3.7
g4 = 1
# Get the clifford algebra for the model we are examining.
cliff = Clifford(p, q)
cliff.introduce()
odd_products = cliff.get_odd_gamma_products()
# odd_products = np.array(odd_products, order='C')
# %%

"""
There are some import hyperparameters for the Metropolis-Hastings algorithm. Specifically,
in our implementation, the variable weightA is important, which is the step size between Dirac operators.
WeightA to be very small, so that the steps between Diracs is relatively small. There is a balance between having a small enough step size so that many Diracs are accepted and we "follow the valley downwards" and having a large enough step size so that we can escape local minima.

This value will change for each type, it seems to be a hyperparameter of this setup, i.e. requires tuning. weightA values for the types that give acceptance_rate/num_moves ~50% (this is what I understand to be a good thing to achieve)

(2,0) -> 1./np.power(cliff.matdim,3)/25
(1,3) -> 1./e-2/6 for size 10x10 - takes 30mins to do 40,000 moves.
"""

# Initialise D: we set the weight A so we start with a much wide range of the parameter space

weightA = 1 / 10  # If we use too large a value here, then the proposed starting Dirac causes the weight action to have a large imaginary component. So start with it ~1
matdim = cliff.matdim
D = DiracOperator.random_Dirac_op(odd_products, N, weightA, cliff.matdim)
# D = D1


@njit(fastmath=True)
def runMonteCarlo(odd_products, D, g2, g4, matdim):
    weightA = 1e-2 / 7
    num_moves = 0
    acceptance_rate = 0
    epochs = 10
    chain_length = 1000
    for i in range(epochs):
        for j in range(chain_length):
            D, acceptance_rate = DiracOperator.update_Dirac(
                D, g2, g4, weightA, acceptance_rate, N, matdim, odd_products)
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

%timeit runMonteCarlo(odd_products, D, g2, g4, matdim)
