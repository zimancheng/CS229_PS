import numpy as np
import matplotlib.pyplot as plt 
import util

from p05b_lwr import LocallyWeightedLinearRegression 

def main(tau_valus, train_path, valid_path, test_path, pred_path):
    """
    Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    
    mse_list = []
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    for tau in tau_valus:
        model = LocallyWeightedLinearRegression(tau=tau)
        model.fit(x_train, y_train)
    
        y_pred = model.predict(x_valid)
        mse = np.mean((y_pred - y_valid)**2)
        mse_list.append(mse)
        print(f"tau: {tau}, validation MSE: {mse}")

        fig, ax = plt.subplots()
        ax.plot(x_train, y_train, 'bx', linewidth=2)
        ax.plot(x_valid, y_pred, 'ro', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(f'output/p05c_{tau}.png')
    
    min_mse = min(mse_list)
    best_tau = tau_valus[np.argmin(mse_list)]
    print(f"The loweset validation MSE is {min_mse} when tau is {best_tau}")
    

