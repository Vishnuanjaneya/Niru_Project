import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load actual and predicted values
    y_true = np.load(r'C:\Users\ADMIN\pythonProject7\y_true.npy')
    y_pred = np.load(r'C:\Users\ADMIN\pythonProject7\y_pred.npy')

    # Create a line plot for actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted Values Over Samples')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
