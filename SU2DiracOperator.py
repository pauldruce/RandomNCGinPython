
""" Random SU2 DIRAC OPERATORS IN NON-COMMUTATIVE GEOMETRY

A random (0,3) SU2 Dirac operator takes the form of:
a\otimes 1 + b\gamma^i \otimes [L_i,.]
where a and b can be any real numbers and L_i are the generators of su(2).

As the only entries that need to explored are the coefficients, this should be a quick simulation to develop.
"""
import numpy as np
import itertools
import cmath

# To be converted from cpp to python.

""" function:  sun_generators

returns the three Lie algebr generators for su2 in the basis [L_1,L_2] = L_3 and cycles.

"""


def su_generators(n):
    j = (n - 1) / 2
    print(j)
    L1 = np.zeros((n, n), dtype=np.complex256)
    L2 = np.zeros((n, n), dtype=np.complex256)
    L3 = np.zeros((n, n), dtype=np.complex256)
    for index in range(n - 1):
        i = index + 1
        L1[index, index + 1] = -1j * (1 / 2) * np.sqrt((j + 1) * (i + (i + 1) - 1) - i * (i + 1))
        L1[index + 1, index] = -1j * (1 / 2) * np.sqrt((j + 1) * (i + (i + 1) - 1) - i * (i + 1))
        L2[index, index + 1] = -1j * -1j * \
            (1 / 2) * np.sqrt((j + 1) * (i + (i + 1) - 1) - i * (i + 1))
        L2[index + 1, index] = -1j * 1j * \
            (1 / 2) * np.sqrt((j + 1) * (i + (i + 1) - 1) - i * (i + 1))
        L3[index, index] = -1j * (j + 1 - i)
    L3[n - 1, n - 1] = -1j * (j + 1 - n)
    return L1, L2, L3


L1, L2, L3 = su_generators(3)
L1
L2
L3
np.dot(L1, L2) - np.dot(L2, L1)

L3


def random_SU2_Dirac_op(odd_products, N, weightA, matdim):
    """ returns a random Dirac operator with entries (a,b) uniformaly sampled between [-1,1].

    The algorithm for producing new Dirac operators calls this function.
    It constructs a random Hermitian matrix

    This is function can be used to initialise the Dirac operator.

    TODO: CHECK THAT THIS RETURNS A NP.MATRIX NOT A NP.ARRAY
    """
    step_size = np.random.normal(-1, 1)
    dirac_dim = matdim * N * N
    D = np.zeros((dirac_dim, dirac_dim), dtype=np.complex128)
    for prod in odd_products:
        temp = random_Hermitian(N)
        if np.array_equal(prod.conj().T, prod) == True:
            Dtemp = anticomm(step_size * temp)
        elif np.array_equal(prod.conj().T, -1 * prod) == True:
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


def update_Dirac(old_D, g2, g4, weightA, acceptance_rate, N, matdim, odd_products):
    """ Updates the Dirac operator according to the Metropolis-Hastings algorithm

    This function inputs the old Dirac operator, the action coupling constants, weightA which dictates the steps size between the old and new Dirac operators, and a variable to calculate the acceptance_rate. This function could be done smoother I'm sure. But it seems to work.

    """
    S_old = action(old_D, g2, g4)
    D = random_Dirac_op(odd_products, N, weightA, matdim)
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
