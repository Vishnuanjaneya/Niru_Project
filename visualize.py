import numpy as np
import matplotlib.pyplot as plt


def main():
    # Load actual and predicted values
    y_true = np.load(r'C:\Users\ADMIN\pythonProject7\y_true.npy')
    y_pred = np.load(r'C:\Users\ADMIN\pythonProject7\y_pred.npy')

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # line for perfect prediction
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
