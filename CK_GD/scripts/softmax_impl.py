
import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, n_classes=2, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_classes = n_classes
        self.weights = None
        self.bias = None
        np.random.seed(random_state)

    def _initialize_parameters(self, n_features):
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot_encode(self, y):
        one_hot = np.zeros((len(y), self.n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        y_encoded = self._one_hot_encode(y)

        for i in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_encoded))
            db = (1 / n_samples) * np.sum(y_pred - y_encoded, axis=0, keepdims=True)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._softmax(linear_model)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return np.argmax(y_proba, axis=1)
