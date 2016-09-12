# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

from itertools import product, combinations
from functools import reduce
from nose.tools import assert_true, assert_raises

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import sparse as sp

from polylearn.kernels import homogeneous_kernel, anova_kernel, safe_power
from polylearn.kernels import _poly_predict, anova_grad


def _product(x):
    return reduce(lambda a, b: a * b, x, 1)


def _power_iter(x, degree):
    return product(*([x] * degree))


def dumb_homogeneous(x, p, degree=2):
    return sum(_product(x[k] * p[k] for k in ix)
               for ix in _power_iter(range(len(x)), degree))


def dumb_anova(x, p, degree=2):
    return sum(_product(x[k] * p[k] for k in ix)
               for ix in combinations(range(len(x)), degree))


def dumb_anova_grad(x, p, degree):
    out = x.copy()

    n_features = p.shape[0]
    for j in range(n_features):
        notj_mask = np.arange(n_features) != j
        out[j] *= dumb_anova(p[notj_mask], x[notj_mask], degree=degree - 1)

    return out


n_samples = 5
n_components = 4
n_features = 7
rng = np.random.RandomState(0)
X = rng.randn(n_samples, n_features)
P = rng.randn(n_components, n_features)
lams = np.array([2, 1, -1, 3])


def test_homogeneous():
    for m in range(1, 5):
        expected = np.zeros((n_samples, n_components))
        for i in range(n_samples):
            for s in range(n_components):
                expected[i, s] = dumb_homogeneous(X[i], P[s], degree=m)
        got = homogeneous_kernel(X, P, degree=m)
        assert_array_almost_equal(got, expected, err_msg=(
            "Homogeneous kernel incorrect for degree {}".format(m)))


def check_anova(degree, method):
    expected = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        for s in range(n_components):
            expected[i, s] = dumb_anova(X[i], P[s], degree=degree)
    got = anova_kernel(X, P, degree=degree, method=method)
    assert_array_almost_equal(got, expected, err_msg=(
        "ANOVA kernel incorrect for degree {} with method {}".format(
            degree,
            method
        )))


def test_anova():
    for degree in (2, 3):
        yield check_anova, degree, 'auto'

    for degree in range(2, 10):
        yield check_anova, degree, 'dp'


def test_anova_ignore_diag_equivalence():
    # predicting using anova kernel
    K = 2 * anova_kernel(X, P, degree=2)
    y_pred = np.dot(K, lams)

    # explicit
    Z = np.dot(P.T, (lams[:, np.newaxis] * P))
    y_manual = np.zeros_like(y_pred)
    for i in range(n_samples):
        x = X[i].ravel()
        xx = np.outer(x, x) - np.diag(x ** 2)
        y_manual[i] = np.trace(np.dot(Z.T, xx))

    assert_array_almost_equal(y_pred, y_manual)


def test_safe_power_sparse():
    # TODO maybe move to a util module or something
    # scikit-learn has safe_sqr but not general power

    X_quad = X ** 4
    # assert X stays sparse
    X_sp = sp.csr_matrix(X)
    for sp_format in ('csr', 'csc', 'coo'):  # not working with lil for now
        X_sp = X_sp.asformat(sp_format)
        X_sp_quad = safe_power(X_sp, degree=4)
        assert_true(sp.issparse(X_sp_quad),
                    msg="safe_power breaks {} sparsity".format(sp_format))
        assert_array_almost_equal(X_quad,
                                  X_sp_quad.A,
                                  err_msg="safe_power differs for {} and "
                                          "dense".format(sp_format))


def test_anova_sparse():
    X_sp = sp.csr_matrix(X)
    for m in (2, 3):
        dense = anova_kernel(X, P, degree=m)
        sparse = anova_kernel(X_sp, P, degree=m)
        assert_array_almost_equal(dense, sparse, err_msg=(
            "ANOVA kernel sparse != dense for degree {}".format(m)))


def test_predict():
    # predict with homogeneous kernel
    y_pred_poly = _poly_predict(X, P, lams, kernel='poly', degree=3)
    K = homogeneous_kernel(X, P, degree=3)
    y_pred = np.dot(K, lams)
    assert_array_almost_equal(y_pred_poly, y_pred,
                              err_msg="Homogeneous prediction incorrect.")

    # predict with homogeneous kernel
    y_pred_poly = _poly_predict(X, P, lams, kernel='anova', degree=3)
    K = anova_kernel(X, P, degree=3)
    y_pred = np.dot(K, lams)
    assert_array_almost_equal(y_pred_poly, y_pred,
                              err_msg="ANOVA prediction incorrect.")


def test_unsupported_degree():
    assert_raises(ValueError, anova_kernel, X, P, degree=1)


def test_unsupported_kernel():
    assert_raises(ValueError, _poly_predict, X, P, lams, kernel='rbf')


def check_anova_grad(degree):
    i = 0
    grad = anova_grad(X, i, P, degree)

    grad_ref = np.empty_like(P)
    for s in range(n_components):
        grad_ref[s] = dumb_anova_grad(X[i], P[s], degree)

    assert_array_almost_equal(grad, grad_ref)


def test_anova_grad():
    for degree in range(2, 10):
        yield check_anova_grad, degree