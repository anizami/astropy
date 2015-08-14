# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Function to calculate the binned flux"""
import numpy as np
cimport numpy as np


def calcbinflux(int array_len, long[:] _indices, long[:] _indices_last,  double[:] avg_y, double[:] _deltax):

#int array_len, np.ndarray _indices, np.ndarray _indices_last,
#                 double[:] avg_y, double[:] _deltax

    cdef long first, last, j
    cdef int i
    cdef long[:] indices_range
    cdef float delta_sum, y_sum

    binned_y = np.empty(shape=array_len, dtype=np.float64)
    for i in range(len(_indices)):
        first = _indices[i]
        last = _indices_last[i]

        indices_range = np.arange(first, last)
        delta_sum = 0.0
        y_sum = 0.0

        for j in indices_range:
            y_sum += (avg_y[j] * _deltax[j])
            delta_sum += _deltax[j]

        if (delta_sum == 0.0):
            raise ZeroDivisionError
        binned_y[i] = y_sum / delta_sum

    return binned_y
