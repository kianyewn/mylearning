# Build sample data
import numpy as np
sample = 9000
n_features = 10
X =  np.random.randn(sample,n_features)

## alternative way to build data
# X = np.hstack([np.random.randn(sample,1) * np.random.randn(1) for _ in range(n_features)])

# insert bias term coefficients
X = np.insert(X, 0, 1, axis=1)

# set the oracle-coefficient. This is what we want to learn
coefs = np.array([1]+ np.linspace(0,1,n_features).tolist())
# get the true labels
y = X.dot(coefs) + np.random.randn(sample)

# Least square analytical solution
trained_coef = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred = X.dot(trained_coef)

## verify with numpy lstsq
m = np.linalg.lstsq(X, y)[0]
assert np.allclose(trained_coef, m) == True

## Plot predictions
# import matplotlib.pyplot as plt
# plt.plot(np.arange(len(y)), y, label='original')
# plt.plot(np.arange(len(y)), y_pred, label='trained')
# plt.legend()
# plt.show()

#################################################################################################################################
## Experiment with pseudo-inverse
## The Moore-Penrose pseudoinverse is a generalized inverse that can be applied to matrices of any rank.
## It's more numerically stable in cases where X is not full rank.
## Generally this is the better approach because real-world datasets may have collinear features or other numerical issues.
## (NOTE): The solution should be the same, but this example with fully independent features led to very bad solution
## (TODO): Investigate why.
#################################################################################################################################
# Formula for SVD:
# if a SVD is applied to a square matrix 𝑀, 𝑀=𝑈𝑆𝑉𝑇, then the inverse of 𝑀 is relatively easy to calculate as 𝑀−1=𝑉𝑆−1𝑈𝑇
# https://math.stackexchange.com/questions/1939962/singular-value-decomposition-and-inverse-of-square-matrix
# gs: svd inverse
# m: (N,N), U: (N, N), S: (N,), V: (N, N)
U, S, V = np.linalg.svd(X.T.dot(X))
S = np.diag(S) # (num_feature, num_feature)
w = V.dot(np.linalg.pinv(S)).dot(U.T).dot(X.T).dot(y)
y_pred_mp =  X.dot(w)

## verify with numpy lstsq
m = np.linalg.lstsq(X, y)[0]
assert np.allclose(w, m) == False


#################################################################################################################################
### Test with online code to make sure that moore-penrose method really does not work
#################################################################################################################################
import math
class l1_regularization():
    """ Regularization for Lasso Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)

class l2_regularization():
    """ Regularization for Ridge Regression """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 *  w.T.dot(w)

    def grad(self, w):
        return self.alpha * w

class l1_l2_regularization():
    """ Regularization for Elastic Net Regression """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w) 
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr) 

class Regression(object):
    """ Base regression model. Models the relationship between a scalar dependent variable y and the independent 
    variables X. 
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N] """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # Gradient of l2 loss w.r.t w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

class LinearRegression(Regression):
    """Linear model.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                            learning_rate=learning_rate)
    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            # X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=10, noise=20)
X = np.insert(X, 0, 1, axis=1)            
            
lr = LinearRegression(gradient_descent=False)
lr.fit(X, y)

## verify with numpy lstsq
m = np.linalg.lstsq(X, y)[0]
assert np.allclose(lr.w,m) == False

####################
### investigated ###
####################
# Reason for the differnece is because my example dataset has some sort of miscalulation
## This is a note to not calculate inverse, as we need to check for invertibility, rank etc etc
### gs: np.linalg.inv vs pinv https://stackoverflow.com/questions/49357417/why-is-numpy-linalg-pinv-preferred-over-numpy-linalg-inv-for-creating-invers
# X=np.matrix([[1,2104,5,1,45],[1,1416,3,2,40],[1,1534,3,2,30],[1,852,2,1,36]])
X=np.matrix([[1,2104,5,],[1,1416,3,],[1,1534,3,],[1,852,2]])
y=np.matrix([[460],[232],[315],[178]])


XT=X.T
XTX=XT@X

pinv=np.linalg.pinv(XTX)
# inv = np.linalg.inv(XTX)
theta_pinv=(pinv@XT)@y

# # calculate inv using pinv.
# Since X is non-square, automatically creates a square matrix X.T.dot(X).dot(X.T)
# theta_inv = np.linalg.pinv(X) @ y
# theta_inv

# calculate inverse using .inv()
theta_inv = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
theta_inv

# correct solution from numpy
m = np.linalg.lstsq(X,y)[0]
m

assert np.allclose(m, theta_pinv) == True
assert np.allclose(m, theta_inv) == True