import numpy as np
from sklearn import datasets
from sklearn.metrics import log_loss



class LogLoss:
    def __init__(self, X, y_true, y_pred):
        self.loss = self.get_log_loss(y_true, y_pred)
        self.grad = self.get_log_loss_grad(X, y_true, y_pred)
        
    def get_log_loss(self, y_true, y_prob):
        loss =  y_true * np.log(y_prob) + (1-y_true) * np.log(1-y_prob)
        return - np.mean(loss)
    
    def get_log_loss_grad(self, X, y_true, y_prob):
        n_samples = y_true.shape[0]

        grad = 1/n_samples * (y_prob- y_true).dot(X)
        return grad
    
    
def sigmoid(x):
    return 1 / (1+ np.exp(-x))
    
assert sigmoid(0) == 0.5

class LogisicRegression:
    def __init__(self, num_features):
        self.w = self.init_weights(num_features)
        self.losses = []

    def init_weights(self, num_features):
        w = np.random.uniform(0,1, size=(num_features)) * 1 / np.sqrt(num_features)
        return w
        
    def fit(self, X, y, learning_rate=1e-3, n_iterations=100):
        for _ in range(n_iterations):
            y_prob = sigmoid(X.dot(self.w))
            log_loss = LogLoss(X, y, y_prob)
            loss = log_loss.loss
            self.losses.append(loss)
            grad = log_loss.grad
            self.w -= learning_rate * grad
            
        return
    def predict(self, X):
        y_prob = sigmoid(X.dot(self.w))
        return y_prob

if __name__ == '__main__':
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    y = np.array([yi if yi < 1 else 1 for yi in y])

 
    lr = LogisicRegression(num_features=X.shape[1])        
    lr.fit(X,y, n_iterations=10000)

    # import matplotlib.pyplot as plt
    # plt.plot(lr.losses)       
    # plt.show()

