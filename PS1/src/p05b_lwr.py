import numpy as np
import matplotlib.pyplot as plt 
import util

from linear_model import LinearModel 

def main(tau, train_path, eval_path):
    """
    Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_valid)
    mse = np.mean((y_pred - y_valid)**2)
    print(f'MSE={mse}')

    fig, ax = plt.subplots()
    ax.plot(x_train, y_train, 'bx', linewidth=2)
    ax.plot(x_valid, y_pred, 'ro', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('output/p05b.png')


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set."""
        self.x = x
        self.y = y

    def predict(self, x):
        """
        Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            W = np.diag(np.exp(-np.sum((self.x - x[i])**2, axis=1) / (2 * self.tau**2)))
            y_pred[i] = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y).T.dot(x[i])

        return y_pred
        
