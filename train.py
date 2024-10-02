import pandas as pd
import numpy as np


def elastic_net(X, y, alpha=1.0, l1_ratio=0.5, num_iterations=1000, learning_rate=0.001):
    """Train the Elastic Net model."""
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for i in range(num_iterations):
        # Calculate model predictions
        model_predictions = np.dot(X, weights) + bias

        # Calculate gradients
        dw = (1 / num_samples) * np.dot(X.T, (model_predictions - y)) + alpha * (
                    l1_ratio * np.sign(weights) + (1 - l1_ratio) * weights)
        db = (1 / num_samples) * np.sum(model_predictions - y)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Print debugging information every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Weights: {weights}, Bias: {bias}, dw: {dw}, db: {db}")

    return weights, bias


def main():
    # Load preprocessed dataset
    DATASET_PATH = r"C:\Users\ADMIN\pythonProject7\data\preprocessed_candidates.csv"
    df = pd.read_csv(DATASET_PATH)

    # Separate features and target variable
    target_column = 'target'
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    # Print shapes and types for debugging
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X data types:\n{X.dtypes}")

    # Train the Elastic Net model
    weights, bias = elastic_net(X.values, y)

    # Save weights, bias, and actual target values
    np.save(r'C:\Users\ADMIN\pythonProject7\weights.npy', weights)
    np.save(r'C:\Users\ADMIN\pythonProject7\bias.npy', bias)
    np.save(r'C:\Users\ADMIN\pythonProject7\y_true.npy', y)

    # Print weights and bias
    print("Weights:", weights)
    print("Bias:", bias)


if __name__ == "__main__":
    main()
