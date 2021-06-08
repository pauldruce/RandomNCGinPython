
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
import numpy as np
import itertools
import cmath


def comm(M):
    """ returns the commutator of a matrix as [M,-] = M x I - I x M.T

    To express the right action of a Matrix onto another, we often write M_n(C) as C^n \otimes (C^n)^*,
    where the * represents the dual vector space. In terms of vectors, if we view C^n as column vectors,
    then (C^n)^* are row vectors, so v^* = v^T. So a matrix acts on the left of an element v \otimes v^*, by Mv \otimes I.v^*,
    where I is the identity matrix. So a matrix acts on the right by I.v \otimes M^T.v^*
    """

    return np.kron(M, np.eye(
        M.shape[0], dtype=np.complex128)) - np.kron(np.eye(M.shape[0], dtype=np.complex128), M.T)


def anticomm(M):
    """ returns the anti-commutator of a matrix as {M,-} = M x I + I x M.T

    To express the right action of a Matrix onto another, we often write M_n(C) as C^n \otimes (C^n)^*,
    where the * represents the dual vector space. In terms of vectors, if we view C^n as column vectors,
    then (C^n)^* are row vectors, so v^* = v^T. So a matrix acts on the left of an element v \otimes v^*, by Mv \otimes I.v^*,
    where I is the identity matrix. So a matrix acts on the right by I.v \otimes M^T.v^*

    So this function returns {M,-} = M \otimes I + I \otimes M.T
    """
    return np.kron(M, np.eye(
        M.shape[0], dtype=np.complex128)) + np.kron(np.eye(M.shape[0], dtype=np.complex128), M.T)


def random_Hermitian(N):
    """
    Creates an NxN element of a Gaussian Hermitian Ensemble
    """
    m = np.random.standard_normal((N, N)) + (
        np.random.standard_normal((N, N)) * 1j)
    m = m + m.conj().T
    return m


def random_Dirac_op(odd_products, N, weightA, matdim):
    """ returns a random Dirac operator with entries uniformaly sampled between [-1,1] + i[-1,1].

    The algorithm for producing new Dirac operators calls this function.
    It constructs a random Hermitian matrix

    This is function can be used to initialise the Dirac operator.

    TODO: CHECK THAT THIS RETURNS A NP.MATRIX NOT A NP.ARRAY
    """
    step_size = np.random.normal(weightA, weightA)
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
