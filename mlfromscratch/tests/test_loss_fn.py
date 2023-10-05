import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'mlfromscratch'))
os.path.join(cwd,'mlfromscratch')

import numpy as np
from sklearn.datasets import make_regression
from sklearn import datasets
from sklearn.metrics import mean_squared_error, log_loss
from linear_regression import MeanSquaredErrorLoss
import torch
import torch.nn as nn

def test_mean_squared_error():
    # Verify manual code
    def custom_mean_squared_error(y, y_hat):
        return np.mean((y-y_hat) **2)

    y = np.random.randn(100)
    y_pred = np.random.randn(100)

    sklearn_e = mean_squared_error(y, y_pred)
    my_e = custom_mean_squared_error(y, y_pred)
    assert np.allclose(my_e, sklearn_e)

def test_mean_squared_error_gradient():
    ##############################
    #### Create data samples #####
    ##############################
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

    
def test_log_loss():
    def custom_log_loss(y_true, y_prob):
        # prob_0 = 1 - y_prob
        loss = - np.mean(y_true * np.log(y_prob) + (1-y_true) * np.log(1-y_prob))
        return loss
    # test with scikit learn 
    y = np.array([1,0,1,0,1,0,1,0,0,0])
    y_prob = np.random.uniform(0, 1, size=(len(y)))
    my_loss = custom_log_loss(y, y_prob)            
    sk_loss = log_loss(y, y_prob)
    # verify log_loss
    assert np.allclose(my_loss, sk_loss)

def test_log_loss_gradient():
    dataset = datasets.load_iris()
    X = dataset.data
    y = dataset.target
    y = np.array([yi if yi < 1 else 1 for yi in y])
    torch.manual_seed(229)
    # logistc regression prediction
    X_pt = torch.tensor(X, dtype=float, requires_grad=False)
    weights = torch.randn(X.shape[-1], dtype=float)
    weights = torch.tensor(weights.clone(), requires_grad=True)
    pred = X_pt @ weights
    sigmoid = nn.Sigmoid()
    prob = sigmoid(pred)

    # get log loss via autograd
    y_pt = torch.tensor(y, dtype=torch.long)
    log_loss = -torch.mean(y_pt * torch.log(prob) + (1-y_pt) * torch.log(1-y_pt))
    log_loss.backward()
    
    # manual calculation of gradinet
    manual_grad = (1 / y_pt.shape[0]) * (prob - y_pt) @ (X_pt)
    assert torch.allclose(weights.grad, manual_grad, atol=0.1)