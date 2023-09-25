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

class Attention(nn.Module):
    def __init__(self, 
                 dim,
                 num_head, 
                 dim_head, 
                 dropout=0):
        super().__init__()
        inner_dim = dim_head * num_head
        self.num_head = num_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q,k,v = self.to_qkv(x).chunk(3, dim=-1)# (B, T, dim) -> (B, T, inner_dim * 3) -> (B, T, inner_dim) * 3
        q,k,v = map(lambda x: x.reshape(B,T, self.num_head, -1).transpose(1,2), [q,k,v]) # (B,T, inner_dim) -> (B, T, H, head_dim) -> (B, H, T, head_dim)
        sim =  q @ k.transpose(-2,-1) # (B, H, T, head_dim), (B, H, T, head_dim) -> (B, H, T, T)
        # normalize for numerical stability
        sim = sim * self.scale
        # normalize to sum to 1
        sim = sim.softmax(dim=-1)     
        # attn dropout    
        sim = self.dropout(sim)
        
        # calc attention
        attn = sim @ v # (B, H, T, T), (B,H, T, head_dim) -> (B, H, T, head_dim)

        # projection back to original dim
        out = attn.transpose(1,2).reshape(B,T, -1) # (B, H, T, head_dim) -> (B, T, head_dim * num_head)
        out = self.to_out(out) # (B, T, head_dim * num_head) -> (B,T, dim)
        # residual dropout
        out = self.dropout(out)
        return out


# Attention
batch_size = 2
num_cat = 4
embd_dim= 5
num_numerical = 3
cat_embd = torch.randn(batch_size, num_cat, embd_dim)

input_dim = embd_dim
num_head = 10
head_dim = 3
inner_dim = num_head * head_dim
ma = Attention(dim=embd_dim, num_head=num_head, dim_head=3)
ma(cat_embd).shape



class EncoderLayer(nn.Module):
    def __init__(self, dim, num_head, dim_head, dropout=0):
        super().__init__()
        self.attn = Residual(PreNorm(fn=Attention(dim=dim,
                                                  num_head=num_head,
                                                  dim_head=dim_head,
                                                  dropout=dropout
                                                  ), 
                                     dim=dim))
        self.ff = Residual(PreNorm(fn=FeedForward(dim=dim),
                                   dim=dim))
        
    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x
    
encoder_layer = EncoderLayer(dim=embd_dim,
                               num_head=num_head,
                               dim_head=head_dim)
out = encoder_layer(cat_embd)
out.shape
