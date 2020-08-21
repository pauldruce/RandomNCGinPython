import numpy as np
import cmath
from scipy import *


def GOE(N):
    '''
    Creates an NxN element of a Guassian Orthogonal Ensemble, by
    creating an NxN matrix of Gaussian random variables.
    This is done by using the random array function
        random.standard_normal([shape])
    with [shape] = (N,N), and adding it to its
    transpose (applying transpose to the matrix)
    '''
    m = random.standard_normal((N, N))
    m = m + np.transpose(m)
    return m


def GOE_Ensemble(num, N):
    ensemble = [GOE(N) for n in range(num)]
    return ensemble


def GUE(N):
    """
    Creates an NxN element of a Gaussian Uniary Ensemble
    """
    m = np.asmatrix(np.random.standard_normal((N, N)) +
                    np.random.standard_normal((N, N)) * 1j)
    m = m + m.H
    return m


def GUE_Ensemble(num, N):
    ensemble = [GUE(N) for n in range(num)]
    return ensemble


ensemble = GUE_Ensemble(3, 40)

ensemble
