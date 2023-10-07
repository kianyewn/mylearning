from sklearn.datasets import make_regression
import numpy as np

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
    
class l1_regularization:
    def __init__(self, l1_lambda=0):
        self.l1_lambda = l1_lambda
        
    def loss(self,  w):
        # return self.l1_lambda * np.linalg.norm(w)
        return self.l1_lambda * np.sum(np.abs(w))

    def grad(self, w):
        return self.l1_lambda * np.sign(w)

class l2_regularization:
    def __init__(self, l2_lambda):
        self.l2_lambda = l2_lambda

    def loss(self, w):
        return self.l2_lambda * np.linalg.norm(w) ** 2
    
    def grad(self, w):
        return self.l2_lambda * 2 * w

class l1_l2_regularization:
    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha 
        self.l1_ratio = l1_ratio
        
    def loss(self, w):
        l1_loss = self.l1_ratio * np.sum(np.abs(w))
        l2_loss = (1-self.l1_ratio) * np.sum(w * w)
        return self.alpha * (l1_loss + l2_loss)
    
    def grad(self,w ):
        l1_grad = self.l1_ratio * np.sign(w)
        l2_grad = (1-self.l1_ratio) * 2 * w
        return self.alpha * (l1_grad + l2_grad)
    
class LinearRegression:
    def __init__(self, n_features, regularizer=None):
        self.w = self.init_weights(n_features)
        self.losses = []
        self.regularizer = regularizer

    def init_weights(self, n_features):
        return np.random.uniform(0,1, size=(n_features)) * 1 / np.sqrt(n_features)
    
    def fit(self, X, y, learning_rate=1e-3, n_iterations=100):
        for _ in range(n_iterations):
            y_pred = X.dot(self.w) # (M,N), (N) -> (M,)
            mse = MeanSquaredErrorLoss(X, y, y_pred)
            loss = mse.loss
            if self.regularizer is not None:
                loss += self.regularizer.loss(self.w)
            self.losses.append(loss)
            grad = mse.grad
            if self.regularizer is not None:
                grad += self.regularizer.grad(self.w)
            self.w -=  learning_rate * grad
        return

    def predict(self, X):
        y_pred = X.dot(self.w)
        return y_pred

if __name__ == '__main__':
    X, y = make_regression(n_samples=100, n_features=10, noise=100)
    lr = LinearRegression(n_features=10)   
    lr.fit(X, y, learning_rate=1e-3, n_iterations=9000) 
    losses_og = lr.losses
    lstsq_sol = np.linalg.lstsq(X,y)[0]
    assert np.allclose(lstsq_sol, lr.w, atol=0.01), f'{lstsq_sol}, {lr.w}'
        
    # l1_regularization
    l1_reg = l1_regularization(l1_lambda=10)
    lr = LinearRegression(n_features=10, regularizer=l1_reg)   
    lr.fit(X, y, learning_rate=1e-3, n_iterations=9000)
    losses_l1= lr.losses 
    
    # l2_regularization
    l2_reg = l2_regularization(l2_lambda=10)
    lr = LinearRegression(n_features=10, regularizer=l2_reg)   
    lr.fit(X, y, learning_rate=1e-3, n_iterations=9000)
    losses_l2 = lr.losses 
    
    # elastic net regularization
    l1_l2_reg = l1_l2_regularization(alpha=10, l1_ratio=0.5)
    lr = LinearRegression(n_features=10, regularizer=l1_l2_reg)   
    lr.fit(X, y, learning_rate=1e-3, n_iterations=9000)
    losses_l1_l2= lr.losses 
    
    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    plt.plot(np.arange(len(losses_og)), losses_og, label='original loss',)
    plt.plot(np.arange(len(losses_og)),losses_l1, label='l1 loss',)
    plt.plot(np.arange(len(losses_og)),losses_l2, label='l2 loss',)
    plt.plot(np.arange(len(losses_og)),losses_l1_l2, label='l1_l2 loss',)
    plt.legend()
    plt.show()
                
    ## addiitonal, plot the numpy prediction for first dimension
    # import matplotlib.pyplot as plt
    # plt.plot(res[:,0], res[:, 1], label='true')
    # plt.plot(res[:,0], res[:, 2], label='pred')
    # plt.legend()
    # plt.show()