import numpy as np


class LinearRegressionElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, lr=0.01, n_iterations=1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(self.n_iterations):
            predictions = X.dot(self.theta)
            errors = predictions - y
            gradient = (X.T.dot(errors) + self.alpha * (
                        self.l1_ratio * np.sign(self.theta) + (1 - self.l1_ratio) * self.theta)) / m
            self.theta -= self.lr * gradient

    def predict(self, X):
        return X.dot(self.theta)


def predict_company(name, age, qualification, skills, experience, df):
    """Predict the company based on input features using preprocessed data."""
    # Prepare the input feature vector based on the input data
    input_data = np.array([[age, experience]])
    input_data = np.hstack([input_data, df.drop('target', axis=1).mean().values])  # Using mean values as placeholders

    # Initialize the model
    model = LinearRegressionElasticNet(alpha=1.0, l1_ratio=0.5)

    # Assuming the target variable is in the last column of df
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Fit the model
    model.fit(X, y)

    # Predict company for the new input
    predicted_company_idx = model.predict(input_data.reshape(1, -1)).round()

    # Map the predicted index to an actual company name (mock implementation)
    company_mapping = {0: 'Company A', 1: 'Company B', 2: 'Company C'}
    return company_mapping.get(int(predicted_company_idx), 'Unknown Company')
