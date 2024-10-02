import numpy as np

class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, lr=0.01, iterations=1000):
        self.alpha = alpha            # Regularization strength (lambda)
        self.l1_ratio = l1_ratio      # Mix between L1 and L2 regularization
        self.lr = lr                  # Learning rate
        self.iterations = iterations  # Number of iterations (epochs)

    def _compute_cost(self, X, y, y_pred, theta):
        m = len(y)
        mse = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        l1_penalty = self.l1_ratio * np.sum(np.abs(theta))
        l2_penalty = (1 - self.l1_ratio) * np.sum(theta ** 2)
        return mse + self.alpha * (l1_penalty + l2_penalty)

    def _compute_gradient(self, X, y, y_pred, theta):
        m = len(y)
        gradient = (1 / m) * X.T.dot(y_pred - y) + \
                   self.alpha * (self.l1_ratio * np.sign(theta) + (1 - self.l1_ratio) * theta)
        return gradient

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)  # Initialize theta to zeros
        y = y.astype(float)       # Ensure y is a float

        # Gradient Descent Loop
        for _ in range(self.iterations):
            y_pred = X.dot(self.theta)
            cost = self._compute_cost(X, y, y_pred, self.theta)
            gradient = self._compute_gradient(X, y, y_pred, self.theta)
            self.theta -= self.lr * gradient

    def predict(self, X):
        return X.dot(self.theta)
