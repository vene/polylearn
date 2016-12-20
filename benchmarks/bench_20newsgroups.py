# Benchmark polynomial classifiers on bag-of-words text classification
# Inspired from: https://github.com/scikit-learn/scikit-learn/blob/master
#                /benchmarks/bench_20newsgroups.py

from time import time

import numpy as np
import scipy.sparse as sp

from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_20newsgroups_vectorized

from polylearn import (FactorizationMachineClassifier,
                       PolynomialNetworkClassifier)


estimators = {
    'fm-2': FactorizationMachineClassifier(beta=0.0001,
                                           n_components=30,
                                           fit_linear=False,
                                           fit_lower=None,
                                           degree=2,
                                           random_state=0,
                                           max_iter=20,
                                           tol=1e-10),

    'polynet-2': PolynomialNetworkClassifier(beta=0.0001,
                                             n_components=15,
                                             degree=2,
                                             fit_lower=None,
                                             max_iter=20,
                                             random_state=0,
                                             tol=1e-10)
}

estimators['fm-3'] = clone(estimators['fm-2']).set_params(degree=3)
estimators['fm-2-ada'] = clone(estimators['fm-2']).set_params(
    solver='adagrad')
estimators['fm-3-ada'] = clone(estimators['fm-3']).set_params(
    solver='adagrad')
estimators['polynet-3'] = (clone(estimators['polynet-2'])
                           .set_params(degree=3, n_components=10))

if __name__ == '__main__':
    data_train = fetch_20newsgroups_vectorized(subset="train")
    data_test = fetch_20newsgroups_vectorized(subset="test")
    X_train_csc = sp.csc_matrix(data_train.data)
    X_test_csc = sp.csc_matrix(data_test.data)
    X_train_csr = sp.csr_matrix(data_train.data)
    X_test_csr = sp.csr_matrix(data_test.data)

    y_train = data_train.target == 0  # atheism vs rest
    y_test = data_test.target == 0

    print("20 newsgroups")
    print("=============")
    print("X_train.shape = {0}".format(X_train_csr.shape))
    print("X_train.dtype = {0}".format(X_train_csr.dtype))
    print("X_train density = {0}"
          "".format(X_train_csr.nnz / np.product(X_train_csr.shape)))
    print("y_train {0}".format(y_train.shape))
    print("X_test {0}".format(X_test_csr.shape))
    print("X_test.dtype = {0}".format(X_test_csr.dtype))
    print("y_test {0}".format(y_test.shape))
    print()

    print("Classifier Training")
    print("===================")
    f1, accuracy, train_time, test_time = {}, {}, {}, {}

    for name, clf in sorted(estimators.items()):
        print("Training %s ... " % name, end="")
        if 'ada' in name:
            X_train, X_test = X_train_csr, X_test_csr
        else:
            X_train, X_test = X_train_csc, X_test_csc
        t0 = time()
        clf.fit(X_train, y_train)
        train_time[name] = time() - t0
        t0 = time()
        y_pred = clf.predict(X_test)
        test_time[name] = time() - t0
        accuracy[name] = accuracy_score(y_test, y_pred)
        f1[name] = f1_score(y_test, y_pred)
        print("done")

    print("Classification performance:")
    print("===========================")
    print()
    print("%s %s %s %s %s" % ("Classifier".ljust(16),
                              "train".rjust(10),
                              "test".rjust(10),
                              "f1".rjust(10),
                              "accuracy".rjust(10)))
    print("-" * (16 + 4 * 11))
    for name in sorted(f1, key=f1.get):
        print("%s %s %s %s %s" % (
            name.ljust(16),
            ("%.4fs" % train_time[name]).rjust(10),
            ("%.4fs" % test_time[name]).rjust(10),
            ("%.4f" % f1[name]).rjust(10),
            ("%.4f" % accuracy[name]).rjust(10)))

    print()
