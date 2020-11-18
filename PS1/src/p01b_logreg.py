import numpy as np
import util
from linear_model import LinearModel 

def main(train_path, eval_path, pred_path):
    """
    Problem 1(b): Logistic regression with Newton's Method.

    Args: 
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predicitions.
    """

    x_train, y_train = util.load_dataset(train_path, add_intercept = True)

    # Train the model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Plot data and decision boundaries
    util.plot(x_train, y_train, model.theta, f"output/p01b_{pred_path[-5]}.png")    # use the number from pred_path

    # Save predictions
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')   # y_pred > 0.5 is classified as 1, otherwise 0
    

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """
        Run Newton's Method to minimizr J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # Init theta
        m, n = x.shape
        self.theta = np.zeros(n)

        # Newton's Method to update theta
        while True:
            theta_prev = np.copy(self.theta)

            # Compute Hessian Matrix
            h_x = 1 / (1 + np.exp(-x.dot(self.theta)))
            H = (x.T * h_x * (1 - h_x)).dot(x) / m
            gradient_J_theta = x.T.dot(h_x - y) / m

            # Update theta
            self.theta -= np.linalg.inv(H).dot(gradient_J_theta)
            
            if np.linalg.norm(self.theta - theta_prev, ord=1) < self.eps:
                break

    def predict(self, x):
        """
        Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return 1 / (1 + np.exp(-x.dot(self.theta)))
