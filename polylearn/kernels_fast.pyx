# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cython.view cimport array
from lightning.impl.dataset_fast cimport RowDataset


# python-facing functions

def _fast_anova_kernel_batch(RowDataset X,
                             double[::1, :] P,
                             unsigned int degree,
                             double[:, ::1] K):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_components = P.shape[0]
    cdef Py_ssize_t n_features = P.shape[1]

    # data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef Py_ssize_t i, s

    # allocate working memory for DP table
    cdef double[::1, :] A = array(shape=(n_features + 1, degree + 1),
                                  itemsize=sizeof(double),
                                  format='d',
                                  mode='fortran')

    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        for s in range(n_components):
            K[i, s] = _fast_anova_kernel(A, P, s, indices, data, n_nz, degree)


def _fast_anova_grad(RowDataset X,
                     Py_ssize_t i,
                     double[::1, :] P,
                     unsigned int degree,
                     double[::1, :] out):

    cdef Py_ssize_t n_components = P.shape[0]
    cdef Py_ssize_t n_features = P.shape[1]

    # data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef Py_ssize_t s

    # allocate working memory for DP table
    cdef double[::1, :] A = array(shape=(n_features + 1, degree + 1),
                                  itemsize=sizeof(double),
                                  format='d',
                                  mode='fortran')

    cdef double[::1, :] Ad = array(shape=(n_features + 2, degree + 2),
                                   itemsize=sizeof(double),
                                   format='d',
                                   mode='fortran')

    X.get_row_ptr(i, &indices, &data, &n_nz)
    for s in range(n_components):
        _fast_anova_kernel_grad(A, Ad, P, s, indices, data, n_nz, degree, out)
