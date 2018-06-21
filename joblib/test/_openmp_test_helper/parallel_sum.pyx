import cython
cimport openmp
from cython.parallel import prange
from libc.stdlib cimport malloc, free


def parallel_sum(int N):
    cdef double Ysum = 0
    cdef int i, num_threads
    cdef double* X = <double *>malloc(N*cython.sizeof(double))

    for i in prange(N, nogil=True):
        num_threads = openmp.omp_get_num_threads()
        Ysum += X[i]

    free(X)
    return num_threads
