import torch.nn as nn
import torch

class Residual(nn.Module):
    """Add original input to the output of the function
    Input shape: (B,T, E) 
    Output shape: (B,T,E)"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)
    
class PreNorm(nn.Module):
    """Apply normalization before the function"""
    def __init__(self, fn, dim):
        super().__init__()
        self.fn = fn
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.ln(x))
    
class FeedForward(nn.Module):
    """
    Input shape: (B,T,E) -> (B,T,E)"""
    def __init__(self, dim, mult=4, dropout=0):
        super().__init__()
        self.ff_net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(),
            nn.Linear(dim * mult, dim), 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff_net(x) 
    
# Attention
batch_size = 2
num_cat = 4
num_numerical=10
embd_dim= 5
num_numerical = 3
cat_embd = torch.randn(batch_size, num_cat, embd_dim)
x_numerical = torch.randn(batch_size, num_numerical)
x_categorical = cat_embd

num_head = 10
head_dim = 3


    
# each batch shares the same linear transformation of the numerical_features
weights = nn.Parameter(torch.randn(num_numerical, embd_dim)) # (T, E)
bias = nn.Parameter(torch.zeros(num_numerical, embd_dim)) # (T, E)
x = torch.randn(2, num_numerical) # (B, T)
embd_x = weights * x.unsqueeze(-1) + bias * x.unsqueeze(-1) # (B,T,E)
embd_x.shape

embd_x[0].shape

manual = weights[0] * x[0][0].unsqueeze(-1) + bias[0] * x[0][0].unsqueeze(-1)
manual.shape


torch.allclose(embd_x[0][0], manu

