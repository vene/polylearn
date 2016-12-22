# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


from libc.math cimport sqrt
from cython cimport view
from lightning.impl.dataset_fast cimport RowDataset

cimport numpy as np
import numpy as np

from .kernels_fast cimport _fast_anova_kernel_grad
from .loss_fast cimport LossFunction


np.import_array()


cdef inline void sync(double* param,
                      unsigned int* last_seen,
                      double grad_norm,
                      double learning_rate,
                      double beta,
                      unsigned int t):
    cdef unsigned int dt
    cdef double sq, correction
    dt = t - last_seen[0]  # dt could be local. is that efficient?
    if dt > 0:
        sq = sqrt(grad_norm)
        correction = sq / (learning_rate * beta + sq + 1e-6)
        param[0] *= correction ** dt
        last_seen[0] = t


cdef inline void ada_update(double* param,
                            double* grad_norm,
                            unsigned int* last_seen,
                            double update,
                            double lp,
                            double learning_rate,
                            double beta,
                            unsigned int t):
    update *= lp

    grad_norm[0] += update ** 2
    cdef double sq = sqrt(grad_norm[0])

    # p <- (p * sq - lr * update) / (lr * beta + sq)
    param[0] *= sq
    param[0] -= learning_rate * update
    param[0] /= 1e-6 + sq + learning_rate * beta
    last_seen[0] = t + 1


def _fast_fm_adagrad(self,
                     double[::1] w,
                     double[::1, :, :] P not None,
                     RowDataset X,
                     double[::1] y not None,
                     unsigned int degree,
                     double alpha,
                     double beta,
                     bint fit_linear,
                     bint fit_lower,
                     LossFunction loss,
                     unsigned int max_iter,
                     double learning_rate,
                     callback,
                     int n_calls):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_components = P.shape[0]
    cdef Py_ssize_t n_features = P.shape[1]

    cdef bint has_callback = callback is not None

    cdef unsigned int it, t
    cdef Py_ssize_t i, s, j, jj, o, order

    cdef double y_pred

    # data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # working memory and DP tables
    # cdef double[:, ::1] P_grad_data
    cdef double[::1, :, :] P_grad_data
    cdef double[::1, :] A
    cdef double[::1, :] Ad

    # to avoid reallocating at every iteration, we allocate more than enough

    P_grad_data = np.empty_like(P, order='f')
    A = np.empty((n_features + 1, degree + 1), order='f')
    Ad = np.empty((n_features + 2, degree + 2), order='f')

    # adagrad bookkeeping, O(2 * n_params)
    cdef double[::1] w_grad_norms
    cdef double[::1, :, :] P_grad_norms
    cdef unsigned int[::1] w_last_seen
    cdef unsigned int[::1, :, :] P_last_seen
    w_grad_norms = np.zeros_like(w)
    P_grad_norms = np.zeros_like(P, order='f')
    w_last_seen = np.zeros_like(w, dtype=np.uint32)
    P_last_seen = np.zeros_like(P, dtype=np.uint32, order='f')

    t = 0
    for it in range(max_iter):

        for i in range(n_samples):
            X.get_row_ptr(i, &indices, &data, &n_nz)

            y_pred = 0

            # catch up
            if fit_linear:
                for jj in range(n_nz):
                    j = indices[jj]
                    sync(&w[j], &w_last_seen[j], w_grad_norms[j],
                         learning_rate, alpha, t)

            for s in range(n_components):
                for jj in range(n_nz):
                    j = indices[jj]
                    sync(&P[s, j, 0], &P_last_seen[s, j, 0],
                         P_grad_norms[s, j, 0], learning_rate, beta, t)

            if fit_lower:
                for order in range(degree - 1, 1, -1):
                    o = degree - order
                    for s in range(n_components):
                        for jj in range(n_nz):
                            j = indices[jj]
                            sync(&P[s, j, o], &P_last_seen[s, j, o],
                                 P_grad_norms[s, j, o], learning_rate,
                                 beta, t)

            # compute predictions
            if fit_linear:
                for jj in range(n_nz):
                    j = indices[jj]
                    y_pred += w[j] * data[jj]

            for s in range(n_components):
                y_pred += _fast_anova_kernel_grad(A,
                                                  Ad,
                                                  P[:, :, 0],
                                                  s,
                                                  indices,
                                                  data,
                                                  n_nz,
                                                  degree,
                                                  P_grad_data[:, :, 0])

            if fit_lower:
                for order in range(degree - 1, 1, -1):
                    o = degree - order
                    for s in range(n_components):
                        y_pred += _fast_anova_kernel_grad(A,
                                                          Ad,
                                                          P[:, :, o],
                                                          s,
                                                          indices,
                                                          data,
                                                          n_nz,
                                                          order,
                                                          P_grad_data[:, :, o])

            # update
            lp = -loss.dloss(y[i], y_pred)

            if fit_linear:
                for jj in range(n_nz):
                    j = indices[jj]
                    ada_update(&w[j],
                               &w_grad_norms[j],
                               &w_last_seen[j],
                               data[jj],  # derivative wrt w[j] is x[j]
                               lp,
                               learning_rate,
                               alpha,
                               t)

            for s in range(n_components):
                for jj in range(n_nz):
                    j = indices[jj]
                    ada_update(&P[s, j, 0],
                               &P_grad_norms[s, j, 0],
                               &P_last_seen[s, j, 0],
                               P_grad_data[s, jj, 0],
                               lp,
                               learning_rate,
                               beta,
                               t)

            if fit_lower:
                for order in range(degree - 1, 1, -1):
                    o = degree - order
                    for s in range(n_components):
                        for jj in range(n_nz):
                            j = indices[jj]
                            ada_update(&P[s, j, o],
                                       &P_grad_norms[s, j, o],
                                       &P_last_seen[s, j, o],
                                       P_grad_data[s, jj, o],
                                       lp,
                                       learning_rate,
                                       beta,
                                       t)

            t += 1
        # end for n_samples

        if has_callback and it % n_calls == 0:
            ret = callback(self, it)
            if ret is not None:
                break
    # end for max_iter

    # finalize
    for j in range(n_features):
        sync(&w[j], &w_last_seen[j], w_grad_norms[j], learning_rate, alpha, t)
    for s in range(n_components):
        for j in range(n_features):
            sync(&P[s, j, 0], &P_last_seen[s, j, 0], P_grad_norms[s, j, 0],
                 learning_rate, beta, t)
    if fit_lower:
        for order in range(degree - 1, 1, -1):
            o = degree - order
            for s in range(n_components):
                for j in range(n_features):
                    sync(&P[s, j, o], &P_last_seen[s, j, o],
                         P_grad_norms[s, j, o], learning_rate,
                         beta, t)
    return it

