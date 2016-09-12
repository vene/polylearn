# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cython.view cimport array
from lightning.impl.dataset_fast cimport RowDataset


cdef inline double _fast_anova_kernel(double[::1, :] A,
                                      double[:, ::1] P,
                                      Py_ssize_t s,
                                      int* indices,
                                      double* data,
                                      int nnz,
                                      unsigned int degree):

    cdef Py_ssize_t t, j, jj

    for jj in range(nnz + 1):
        A[jj, 0] = 1

    for t in range(1, degree + 1):
        for jj in range(0, t):
            A[jj, t] = 0

        for jj in range(t, nnz + 1):
            j = indices[jj - 1]
            A[jj, t] = P[s, j]
            A[jj, t] *= data[jj - 1]
            A[jj, t] *= A[jj - 1, t - 1]
            A[jj, t] += A[jj - 1, t]

    return A[nnz, degree]


cdef inline double _fast_anova_kernel_grad(double[::1, :] A,
                                           double[::1, :] Ad,
                                           double[:, ::1] P,
                                           Py_ssize_t s,
                                           int* indices,
                                           double* data,
                                           int nnz,
                                           unsigned int degree,
                                           double[:, ::1] out):


    # computing the kernel value has the side effect of filling up A
    cdef double val = _fast_anova_kernel(A, P, s, indices, data, nnz, degree)
    cdef Py_ssize_t t, j, jj

    # initialize memory
    for t in range(0, degree + 1):
        for jj in range(0, nnz + 1):
            Ad[jj, t] = 0

    Ad[nnz, degree] = 1

    for t in range(degree, 0, -1):
        for jj in range(nnz, t - 1, -1):
            if jj < nnz:
                if t < degree:
                    j = indices[jj]
                    Ad[jj, t] = Ad[jj + 1, t + 1]
                    Ad[jj, t] *= P[s, j]
                    Ad[jj, t] *= data[jj]

                Ad[jj, t] += Ad[jj + 1, t]


    for jj in range(nnz):
        out[s, jj] = 0
        for t in range(degree):
            out[s, jj] += Ad[jj + 1, t + 1]  * A[jj, t] * data[jj]

    return val


# python-facing functions

def _fast_anova_kernel_batch(RowDataset X,
                             double[:, ::1] P,
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
                     double[:, ::1] P,
                     unsigned int degree,
                     double[:, ::1] out):

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

    cdef double[::1, :] Ad = array(shape=(n_features + 1, degree + 1),
                                   itemsize=sizeof(double),
                                   format='d',
                                   mode='fortran')

    X.get_row_ptr(i, &indices, &data, &n_nz)
    for s in range(n_components):
        _fast_anova_kernel_grad(A, Ad, P, s, indices, data, n_nz, degree, out)
