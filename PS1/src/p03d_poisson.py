import numpy as np
import matplotlib.pyplot as plt 
import util

from linear_model import LinearModel

def main(lr, train_path, eval_path, pred_path):
    """
    Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    model = PoissonRegression(step_size = lr)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred)

    fig, ax = plt.subplots()
    ax.plot(y_eval, y_pred, 'go')
    ax.set_xlabel('true counts')
    ax.set_ylabel('estimate counts')
    plt.savefig('output/p03d_ratio.png')



class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """
        Run gradient ascent to maximize log-likelihood for Poisson Regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape
        self.theta = np.zeros(n)

        while True:
            theta = np.copy(self.theta)
            self.theta += self.step_size * x.T.dot(y - np.exp(x.dot(self.theta))) / m

            if np.linalg.norm(self.theta - theta, ord = 1) < self.eps:
                break

    def predict(self, x):
        """
        Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).
        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        return np.exp(x.dot(self.theta))