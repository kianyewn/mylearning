import numpy as np
from sklearn import datasets

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
    
class LogisicRegression:
    def __init__(self, num_features, regularizer=None):
        self.w = self.init_weights(num_features)
        self.losses = []
        self.regularizer = regularizer

    def init_weights(self, num_features):
        w = np.random.uniform(0,1, size=(num_features)) * 1 / np.sqrt(num_features)
        return w
        
    def fit(self, X, y, learning_rate=1e-3, n_iterations=100):
        for _ in range(n_iterations):
            y_prob = sigmoid(X.dot(self.w))
            log_loss = LogLoss(X, y, y_prob)
            loss = log_loss.loss
            if self.regularizer:
                loss += self.regularizer.loss(self.w)
            self.losses.append(loss)
            grad = log_loss.grad
            if self.regularizer:
                grad += self.regularizer.grad(self.w)
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
    losses_og = lr.losses

    # l1_regularization
    l1_reg = l1_regularization(l1_lambda=10)
    lr = LogisicRegression(num_features=X.shape[1], regularizer=l1_reg)   
    lr.fit(X, y, learning_rate=1e-3, n_iterations=9000)
    losses_l1= lr.losses 
    
    # l2_regularization
    l2_reg = l2_regularization(l2_lambda=10)
    lr = LogisicRegression(num_features=X.shape[1], regularizer=l2_reg)   
    lr.fit(X, y, learning_rate=1e-3, n_iterations=9000)
    losses_l2 = lr.losses 
    
    # elastic net regularization
    l1_l2_reg = l1_l2_regularization(alpha=10, l1_ratio=0.5)
    lr = LogisicRegression(num_features=X.shape[1], regularizer=l1_l2_reg)   
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
