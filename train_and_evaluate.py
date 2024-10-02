import pandas as pd
import numpy as np

# Load preprocessed data
data_path = "C:\\Users\\ADMIN\\pythonProject7\\data\\preprocessed_candidates.csv"
data = pd.read_csv(data_path)

# Prepare features (X) and target (y)
X = data.drop(columns=['target']).values
y = data['target'].values

# Feature scaling (Standardization)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std


# Define Elastic Net
def elastic_net(X, y, learning_rate=0.0001, n_iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0

    for i in range(n_iterations):
        model_predictions = np.dot(X, weights) + bias

        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (model_predictions - y))
        db = (1 / n_samples) * np.sum(model_predictions - y)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

        if i % 100 == 0:  # Print every 100 iterations
            print(f"Iteration {i}: Weights: {weights}, Bias: {bias}, dw: {dw}, db: {db}")

    return weights, bias


# Train the model
weights, bias = elastic_net(X, y)

# Make predictions
y_pred = np.dot(X, weights) + bias

# Calculate Mean Squared Error
mse = np.mean((y_pred - y) ** 2)
print(f"Mean Squared Error: {mse}")
