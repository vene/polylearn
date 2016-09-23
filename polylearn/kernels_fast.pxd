# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


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
    for t in range(0, degree + 2):
        for jj in range(0, nnz + 2):
            Ad[jj, t] = 0

    Ad[nnz, degree] = 1

    for t in range(degree, 0, -1):
        for jj in range(nnz - 1, t - 1, -1):
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
