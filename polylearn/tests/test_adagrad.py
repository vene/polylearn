from nose.tools import assert_less_equal

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises_regex

import scipy.sparse as sp

from sklearn.metrics import mean_squared_error

from polylearn.kernels import _poly_predict
from polylearn import FactorizationMachineRegressor
from .test_kernels import dumb_anova_grad


def sg_adagrad_slow(P, X, y, degree, beta, max_iter, learning_rate,
                    scale_regularization=True):

    n_samples = X.shape[0]
    n_components = P.shape[0]

    grad_norms = np.zeros_like(P)

    if not scale_regularization:
        beta /= 0.5 * n_samples

    for it in range(max_iter):

        for i in range(n_samples):
            x = X[i]
            y_pred = _poly_predict(np.atleast_2d(x), P, np.ones(n_components),
                                   kernel='anova', degree=degree)

            for s in range(n_components):
                update = dumb_anova_grad(x, P[s], degree)
                update *= y_pred - y[i]

                grad_norms[s] += update ** 2

                P[s] = P[s] * np.sqrt(grad_norms[s]) - learning_rate * update
                P[s] /= 1e-6 + np.sqrt(grad_norms[s]) + learning_rate * beta

    return P


n_components = 3
n_features = 15
n_samples = 20

rng = np.random.RandomState(1)

X = rng.randn(n_samples, n_features)
P = rng.randn(n_components, n_features)

lams = np.ones(n_components)


class LossCallback(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.objectives_ = []

    def __call__(self, fm, it):

        # temporarily reshuffle fm.P_ to ensure predict works
        old_P = fm.P_
        fm.P_ = np.transpose(old_P, [2, 0, 1])
        y_pred = fm.predict(self.X)
        fm.P_ = old_P

        obj = ((y_pred - self.y) ** 2).mean()
        obj += fm.alpha * (fm.w_ ** 2).sum()
        obj += fm.beta * (fm.P_ ** 2).sum()
        self.objectives_.append(obj)


class CheckChangeCallback(object):
    def __init__(self):
        self.old_P = None

    def __call__(self, fm, it):
        if self.old_P is not None:
            diff = np.sum((self.old_P - fm.P_) ** 2)
            assert_less_equal(1e-8, diff)
        self.old_P = fm.P_.copy()


def check_adagrad_decrease(degree):
    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)

    cb = LossCallback(X, y)
    est = FactorizationMachineRegressor(degree=degree, n_components=3,
                                        fit_linear=True, fit_lower=None,
                                        solver='adagrad',
                                        init_lambdas='ones',
                                        max_iter=100,
                                        learning_rate=0.01,
                                        beta=1e-8,
                                        callback=cb,
                                        n_calls=1,
                                        random_state=0)
    est.fit(X, y)
    # obj = np.array(cb.objectives_)
    # assert_array_less(obj[1:], obj[:-1])


def test_adagrad_decrease():
    for degree in range(2, 6):
        yield check_adagrad_decrease, degree


def check_adagrad_fit(degree):
    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)

    est = FactorizationMachineRegressor(degree=degree, n_components=5,
                                        fit_linear=True, fit_lower=None,
                                        solver='adagrad',
                                        init_lambdas='ones',
                                        max_iter=2000,
                                        learning_rate=0.25,
                                        alpha=1e-10,
                                        beta=1e-10,
                                        random_state=0)

    est.fit(X, y)
    y_pred = est.predict(X)
    err = mean_squared_error(y, y_pred)

    assert_less_equal(err, 1e-3,
        msg="Error {} too big for degree {}.".format(err, degree))


def test_adagrad_fit():
    for degree in range(2, 6):
        yield check_adagrad_fit, degree


def check_adagrad_same_as_slow(degree, sparse):

    beta = 1e-5
    lr = 0.01

    if sparse:
        this_X = X.copy()
        this_X[np.abs(this_X) < 1] = 0
        this_X_sp = sp.csr_matrix(this_X)
    else:
        this_X = this_X_sp = X

    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)

    P_fast = np.random.RandomState(42).randn(1, P.shape[0], P.shape[1])
    P_slow = P_fast[0].copy()

    reg = FactorizationMachineRegressor(degree=degree, n_components=P.shape[0],
                                        fit_lower=None, fit_linear=False,
                                        solver='adagrad', init_lambdas='ones',
                                        beta=beta, warm_start=True,
                                        max_iter=2, learning_rate=lr,
                                        random_state=0)
    reg.P_ = P_fast
    reg.fit(this_X_sp, y)

    P_slow = sg_adagrad_slow(P_slow, this_X, y, degree, beta=beta, max_iter=2,
                             learning_rate=lr)

    assert_array_almost_equal(reg.P_[0, :, :], P_slow)


def test_adagrad_same_as_slow():
    for sparse in (False, True):
        for degree in range(2, 5):
            yield check_adagrad_same_as_slow, degree, sparse


def test_callback_P_change():
    # Check that the learner actually updates self.P_ on the fly.
    # Otherwise the callback is pretty much useless
    y = _poly_predict(X, P, lams, kernel="anova", degree=4)
    cb = CheckChangeCallback()
    reg = FactorizationMachineRegressor(degree=4, solver='adagrad',
                                        callback=cb, n_calls=1, max_iter=3,
                                        random_state=0)
    reg.fit(X, y)


def test_predict_sensible_error():
    y = _poly_predict(X, P, lams, kernel="anova", degree=4)
    reg = FactorizationMachineRegressor(degree=4, solver='adagrad',
                                        fit_linear=False, fit_lower=None,
                                        max_iter=3, random_state=0)
    reg.fit(X, y)
    assert_raises_regex(ValueError,
                        "Incompatible dimensions",
                        reg.predict,
                        X[:, :2])
    reg.P_ = np.transpose(reg.P_, [1, 2, 0])
    assert_raises_regex(ValueError, "wrong order", reg.predict, X)


