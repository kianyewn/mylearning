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

# n_samples must be small so that the impact on the floating point differences do not affect the test
X, y = make_regression(n_samples=2, n_features=10, noise=100)

# originally thought this will disperse floating point differences. But no. Floating point differences come up due to number of samples (when we sum up the gradients across the samples), not magnitude of features
# X = X * 99999999999999 # originally thought this will disperse floating point differences. But no. Floating point differences come up due to number of samples (when we sum up the gradients across the samples), not magnitude of features

X_coef = np.linalg.lstsq(X, y)[0]
y_pred = X.dot(X_coef)



mse = MeanSquaredErrorLoss(X, y, y_pred)
loss = mse.loss
grad = mse.grad

# Verify gradient logic with pytorch. 
# Recode everything from scratch in pytorch
import torch
import torch.nn as nn

torch_X = torch.tensor(X, requires_grad=True)
torch_X.transpose(0,1).shape
torch_y = torch.tensor(y, requires_grad=True)
torch_coef = torch.linalg.lstsq(torch_X, torch_y)[0] # torch.linalg.inv(torch_X.transpose(0,1) @torch_X) @ torch_X.transpose(0,1) @ torch_y # torch.tensor(y_pred, requires_grad=True)
torch_coef = torch.tensor(torch_coef.clone(), requires_grad=True)
torch_y_pred = torch_X @ torch_coef
loss = torch.mean((torch_y-torch_y_pred)**2)

# gradient through torch autograd
loss.backward()

# gradient via analytical solution
torch_grad_manual = - 2 * 1 / torch_X.shape[0] * (torch_y - torch_y_pred) @ torch_X

# confirm gradient from analytical solution is same as torch autograd
assert torch.allclose(torch_coef.grad, torch_grad_manual)

# have to detach from torch due to numerical differences when converting X to torch.tensor!
X = torch_X.detach().numpy()
y = torch_y.detach().numpy()
y_pred = torch_y_pred.detach().numpy()
mse = MeanSquaredErrorLoss(X, y, y_pred)
loss = mse.loss
grad = mse.grad
assert torch.allclose(torch_grad_manual, torch.tensor(grad))

## addiitonal, plot the numpy prediction for first dimension
# import matplotlib.pyplot as plt
# plt.plot(res[:,0], res[:, 1], label='true')
# plt.plot(res[:,0], res[:, 2], label='pred')
# plt.legend()
# plt.show()