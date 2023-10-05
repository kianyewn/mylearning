from sklearn.datasets import make_regression
import numpy as np
from sklearn.metrics import mean_squared_error

## Implement from scratch MSE loss
class MeanSquaredErrorLoss:
    def __init__(self, X, y_true, y_pred):
        self.loss = self.get_mean_squared_error(y_true, y_pred)
        self.grad = self.get_mean_squared_error_grad(y_true, y_pred, X)

    def get_mean_squared_error(self, y_true, y_pred):
        loss = np.mean((y_true-y_pred)**2)
        return loss
    
    def get_mean_squared_error_grad(self, y_true, y_pred, X):
        n_samples = y_true.shape[0]
        grad1 =  - 2 * 1/n_samples * (y_true-y_pred).dot(X) # (m,) , (m, n) -> (n)
        grad =  - 2 * 1/n_samples * X.T.dot(y_true-y_pred)
        assert np.allclose(grad1, grad)
        return grad
    
    
class LinearRegression:
    def __init__(self, n_features):
        self.w = self.init_weights(n_features)
        self.losses = []

    def init_weights(self, n_features):
        return np.random.uniform(0,1, size=(n_features)) * 1 / np.sqrt(n_features)
    
    def fit(self, X, y, learning_rate=1e-3, n_iterations=100):

        for _ in range(n_iterations):
            y_pred = X.dot(self.w) # (M,N), (N) -> (M,)
            mse = MeanSquaredErrorLoss(X, y, y_pred)
            loss = mse.loss
            self.losses.append(loss)
            grad = mse.grad
            self.w -=  learning_rate * grad
        return

    def predict(self, X):
        y_pred = X.dot(self.w)
        return y_pred
if __name__ == '__main__':
    X, y = make_regression(n_samples=100, n_features=10, noise=100)
    lr = LinearRegression(n_features=10)   
    lr.fit(X, y, learning_rate=1e-3, n_iterations=5000) 

    # import matplotlib.pyplot as plt
    # plt.plot(lr.losses)
    # plt.show()
                
    lstsq_sol = np.linalg.lstsq(X,y)
    assert np.allclose(lstsq_sol, lr.w)
        

    ## addiitonal, plot the numpy prediction for first dimension
    # import matplotlib.pyplot as plt
    # plt.plot(res[:,0], res[:, 1], label='true')
    # plt.plot(res[:,0], res[:, 2], label='pred')
    # plt.legend()
    # plt.show()