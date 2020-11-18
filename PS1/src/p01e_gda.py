import numpy as np
import util

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """
    Problem 1(e): Gaussian dicriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False) # intercept has to be F, we need x to be m*n to calculate sigma

    # Train GDA
    model = GDA()
    model.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, f'output/p01e_{pred_path[-5]}.png')

    # Save predicitons
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True) # here add intercept since theta is n+1 * 1
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')


class GDA(LinearModel):
    def fit(self, x, y):
        """
        Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape
        self.theta = np.zeros(n + 1)

        # Calculate parameters of exponential family
        y_1 = sum(y == 1)
        phi = y_1 / m
        mu_0 = np.sum(x[y == 0], axis = 0) / (m - y_1)
        mu_1 = np.sum(x[y == 1], axis = 0) / y_1
        sigma = ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)) / m

        # Compute theta
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = 0.5 * (mu_0 + mu_1).T.dot(sigma_inv).dot(mu_0 - mu_1) - np.log((1 - phi) / phi) 
        self.theta[1:] = sigma_inv.dot(mu_1 - mu_0)
        
        # Return theta
        return self.theta 


    def predict(self, x):
        """
        Make a prediction given new inputs x.

        Args: 
            x: Inputs of shape (m, n).
        Returns:
            Outputs of shape (m,).
        """
        return 1 / (1 + np.exp(-x.dot(self.theta))) # notice here x has the intercept term, hence m*n+1 