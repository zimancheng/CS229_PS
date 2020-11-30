import util
import numpy as np
import matplotlib.pyplot as plt 


def calc_grad(X, Y, theta):
    """Compute the gradient of th loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def logistic_regression(X, Y):
    """Train a logistic regression model"""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 1

    i = 0   # iteration count
    while True:
        i += 1
        theta_prev = np.copy(theta)
        grad = calc_grad(X, Y, theta)

        # # learning rate decay 
        # learning_rate /= i**2

        theta -= learning_rate * grad
        if i % 10000 == 0:  # print out iteration detail when it's kth 10000 iterations
            print('Finished %d iterations' % i)
            print('Training loss = %f' % np.mean(np.log(1 + np.exp(-Y * X.dot(theta)))))
            print('||theta_k - theta_k-1|| = %f' % np.linalg.norm(theta - theta_prev, ord=1))

        if np.linalg.norm(theta - theta_prev, ord = 1) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    
    return theta

if __name__=='__main__':
    # Plot dataset A and B
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)

    print('==== Training model on data set A ====')
    theta_a = logistic_regression(Xa, Ya)
    util.plot(Xa, (Ya == 1).astype(int), theta_a, 'output/ds1_a_lr.png')
    
    print('==== Training model on data set B ====')
    theta_b = logistic_regression(Xb, Yb)
    util.plot(Xb, (Yb == 1).astype(int), theta_b, 'output/ds1_b_lr.png')
    
    # dataset B does not converge
    # plot dataset B
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)
    plt.figure()
    util.plot_points(Xb, (Yb == 1).astype(int))
    plt.savefig('output/ds1_b.png')    


