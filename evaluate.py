import pandas as pd
import numpy as np


def elastic_net(X, y, weights, bias, alpha=1.0, l1_ratio=0.5, num_iterations=1000, learning_rate=0.001):
    """Evaluate the Elastic Net model."""
    num_samples = X.shape[0]

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

    return model_predictions


def main():
    # Load preprocessed dataset
    DATASET_PATH = r"C:\Users\ADMIN\pythonProject7\data\preprocessed_candidates.csv"
    df = pd.read_csv(DATASET_PATH)

    # Separate features and target variable
    target_column = 'target'
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    # Load weights and bias
    weights = np.load(r'C:\Users\ADMIN\pythonProject7\weights.npy')
    bias = np.load(r'C:\Users\ADMIN\pythonProject7\bias.npy')

    # Make predictions
    y_pred = elastic_net(X.values, y, weights, bias)

    # Calculate Mean Squared Error
    mse = np.mean((y_pred - y) ** 2)
    print("Mean Squared Error:", mse)

    # Save predictions
    np.save(r'C:\Users\ADMIN\pythonProject7\y_pred.npy', y_pred)


if __name__ == "__main__":
    main()
