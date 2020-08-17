
""" DIRAC OPERATORS IN NON-COMMUTATIVE GEOMETRY

The Dirac operator is constructure from the odd products of gamma matrices, that are tensored
with either a commutator or an anti-commutator.
If the gamma product is hermitian, then it is tensored with a (anti-)commutator of a Hermitian
matrix, and if it is anti-hermitian then it is tensored with a (anti-)commutator of an anti-Hermitian

======== This is not implement here ========
We can then borrow a factor of i from the anti-Hermitian matrices and combine it with the anti-hermitian
gamma product -> making it hermitian.
============================================
So we need to produce the odd gamma products and then construct an operator from tensor products with commutators and anti-commutators.

This is a naive implementation at the moment, where I update D direcly and calculate the action directly with D. As D is a much larger matrix than the H and L matrices that enter to commutators. There is a way to speed up by calculating the action (by hand) in terms of H and L.

This is done in other implementations of NCG, specifically the algorithms used by Dr Lisa Glaser and Mauro D'arcangelo.

The limitation is that the actions get more and more complicated to code for the larger Clifford types.
"""

import time
from CliffordAlgebras import *
import itertools
from pprint import pprint


def get_odd_gamma_products(cliff):
    """ Produce all the odd gamma products

    The function inputs the cliff object (which contains the generators) for a given
    Clifford module, it then extracts the uses these to calculate the necessary products
    """

    k = cliff.n - 1
    # Number of odd products (we are doing n choose r for all odd r)
    number = int(2**k)
    # odd_products = np.zeros(number)
    odd_products = []
    # The plus one is to ensure that if there are an odd num of generators, then the prod of them all is included
    for i in range(1, cliff.n, 2):
        perm = np.array(list(itertools.combinations(cliff.generators, i)))
        for x in perm:
            if len(x) > 1:
                odd_products.append(np.asmatrix(np.linalg.multi_dot(x)))
            else:
                odd_products.append(np.asmatrix(x))
    return odd_products


def comm(M):
    """ returns the commutator of a matrix as [M,-] = M x I - I x M.T

    To express the right action of a Matrix onto another, we often write M_n(C) as C^n \otimes (C^n)^*,
    where the * represents the dual vector space. In terms of vectors, if we view C^n as column vectors,
    then (C^n)^* are row vectors, so v^* = v^T. So a matrix acts on the left of an element v \otimes v^*, by Mv \otimes I.v^*,
    where I is the identity matrix. So a matrix acts on the right by I.v \otimes M^T.v^*
    """
    return np.kron(M, np.identity(M.shape[0])) - np.kron(np.identity(M.shape[0]), M.T)


def anticomm(M):
    """ returns the anti-commutator of a matrix as {M,-} = M x I + I x M.T

    To express the right action of a Matrix onto another, we often write M_n(C) as C^n \otimes (C^n)^*,
    where the * represents the dual vector space. In terms of vectors, if we view C^n as column vectors,
    then (C^n)^* are row vectors, so v^* = v^T. So a matrix acts on the left of an element v \otimes v^*, by Mv \otimes I.v^*,
    where I is the identity matrix. So a matrix acts on the right by I.v \otimes M^T.v^*

    So this function returns {M,-} = M \otimes I + I \otimes M.T
    """
    return np.kron(M, np.identity(M.shape[0])) + np.kron(np.identity(M.shape[0]), M.T)


def random_Hermitian(N):
    """
    Creates an NxN element of a Gaussian Hermitian Ensemble
    """
    m = np.asmatrix(np.random.standard_normal((N, N)) +
                    np.random.standard_normal((N, N)) * 1j)
    m = m + m.H
    return m


def random_Dirac_op(odd_products, N, weightA):
    """ returns a random Dirac operator with entries uniformaly sampled between [-1,1] + i[-1,1].

    The algorithm for producing new Dirac operators calls this function.
    It constructs a random Hermitian matrix

    This is function can be used to initialise the Dirac operator.

    TODO: CHECK THAT THIS RETURNS A NP.MATRIX NOT A NP.ARRAY
    """
    step_size = np.random.normal(weightA, weightA)
    dirac_dim = cliff.matdim * N * N
    D = np.zeros((dirac_dim, dirac_dim))
    for prod in odd_products:
        temp = random_Hermitian(N)
        if np.all(prod.H == prod) == True:
            Dtemp = anticomm(step_size * temp)
        elif np.all(prod.H == -1 * prod) == True:
            Dtemp = comm(step_size * complex(0, 1) * temp)
        Dtemp2 = np.kron(prod, Dtemp)
        D = D + Dtemp2
    return D


def action(D, g2, g4):
    """ calculates the action using the Dirac operators.

    This method of evalutation is slow as it requires dealing with large matrices. But it has the
    advantage of working for any Clifford type with no further calculation. It will just be pretty slow at the moment. This could change with the use of Numba or similar, but needs work.

    TODO: NEED TO CHECK THAT THIS ACTUALLY DOES THE MATRIX MULT. IF NOT CHANGE FOR NUMPY ROUTINE
    """
    if g2 != 0:
        trD2 = np.trace(D * D)
    else:
        trD2 = 0

    if g4 != 0:
        trD4 = np.trace(D * D * D * D)
    else:
        trD4 = 0

    action = g2 * trD2 + g4 * trD4
    # print(action)
    if np.imag(action) < 10e-8:
        return np.real(action)
    else:
        raise ValueError("Your action wasn't real!")


def update_Dirac(old_D, g2, g4, weightA, acceptance_rate):
    """ Updates the Dirac operator according to the Metropolis-Hastings algorithm

    This function inputs the old Dirac operator, the action coupling constants, weightA which dictates the steps size between the old and new Dirac operators, and a variable to calculate the acceptance_rate. This function could be done smoother I'm sure. But it seems to work.

    """
    S_old = action(old_D, g2, g4)
    D = random_Dirac_op(odd_products, N, weightA)
    S_new = action(old_D + D, g2, g4)
    # I think of rand_press as the random pressure which pushes you out of sufficiently small local minima.
    rand_press = np.exp(S_old - S_new)
    # p is the random value between 0 and 1 used in Metropolis-Hastings algorithm
    p = np.random.uniform(0, 1)

    # This is my understanding of the Metropolis-Hastings algorithm
    if S_new < S_old or rand_press > p:
        acceptance_rate = acceptance_rate + 1
        return old_D + D, acceptance_rate
    else:
        return old_D, acceptance_rate


# %%
""" This section could be used in a separate file, but its here for now """
# Some dummy parameters for use when testing:
p = 3
q = 1
N = 10
g2 = -3.7
g4 = 1

# Get the clifford algebra for the model we are examining.
cliff = Clifford(p, q)
cliff.introduce()

# Seed the random generator with the time current time, so we always get new random numbers
t_raw = int(time.time() * 1000.0)
t_str = list(str(t_raw))[5:]
t = int("". join(t_str))
np.random.seed(t)
odd_products = get_odd_gamma_products(cliff)

# Initialise D: we set the weight A so we start with a much wide range of the parameter space
start_time = time.time()
weightA = 1 / 10  # If we use too large a value here, then the proposed starting Dirac causes the weight action to have a large imaginary component. So start with it ~1

D = random_Dirac_op(odd_products, N, weightA)
# D = D1
# D

"""
# We then set the variable weightA to be very small, so that the steps between Diracs is relatively small. There is a balance between having a small enough step size so that many Diracs are accepted and we "follow the valley downwards" and having a large enough step size so that we can escape local minima.

This value will change for each type, it seems to be a hyperparameter of this setup, i.e. requires tuning.

weightA values for the types that give acceptance_rate/num_moves ~50%

(2,0) -> 1./np.power(cliff.matdim,3)/25
(1,3) -> 1./e-2/6 for size 10x10 - takes 15mins to do 20,000 moves.
"""

weightA = 1e-2 / 6
print(weightA)
num_moves = 0
acceptance_rate = 0
for i in range(20):
    for j in range(1000):
        # Run the algorithm for 1000 steps
        D, acceptance_rate = update_Dirac(D, g2, g4, weightA, acceptance_rate)
        num_moves += 1
    # Every 1000s steps, we investigate a little. We print the accepted rate and the action of the Dirac at this moment.
    print(acceptance_rate, num_moves, acceptance_rate / num_moves)
    print(action(D, g2, g4))
    # np.linalg.eigvals(D)
print("--- %s seconds ---" % (time.time() - start_time))
# It would be helpful to store the last Dirac, so we can "continue" the simulation after it reaches 20,0000 steps. For instance, for the type (1,3), 20,000 steps doesnt seem to be enough.
# We can then "initialise" the Dirac with the last Dirac operator from the simulation.
D1 = D


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
