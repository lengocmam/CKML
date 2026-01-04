
import numpy as np
import pandas as pd 

class ManualGaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]

        self.mean = np.zeros((len(self.classes), n_features))
        self.var = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.var[i, :] = X_c.var(axis=0) + self.var_smoothing
            self.priors[i] = X_c.shape[0] / float(len(X))

    def _calculate_likelihood(self, class_idx, X):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(X - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihoods = np.log(self._calculate_likelihood(i, X) + 1e-15)
            posterior = prior + np.sum(likelihoods, axis=1)
            posteriors.append(posterior)
        return self.classes[np.argmax(np.array(posteriors), axis=0)]


def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    n_samples = len(X)
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    test_count = int(n_samples * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test
