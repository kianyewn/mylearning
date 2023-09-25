import torch
import torch.nn as nn
from torch import einsum
from sequences.tabtransformer import (Residual,
                                      FeedForward,
                                      PreNorm,
                                      Attention,
                                      MLP,
                                      TabTransformer
                                      )
from einops import rearrange

def test_residual():
    x = torch.randn(2,3)

    fn = nn.Linear(3,3)
    res = Residual(fn=fn)
    res_out = res(x.clone())
    
    # manual
    res = fn(x.clone()) + x.clone()
    assert torch.allclose(res, res_out)
    
def test_layer_norm():
    # via pytorch
    x = torch.randn(2,3,4)
    ln = nn.LayerNorm(4)
    x_ln = ln(x)
    
    # manaul calculation
    mean = x.mean(dim=-1, keepdim=True) # (B, T, E) -> (B,T,1)
    mean_x2 = (x**2).mean(dim=-1, keepdim=True) # (B,T,E) -> (B,T,1)
    var = mean_x2 - mean**2 # (B,T,1), (B,T,1) -> (B,T,1)
    
    x_norm = (x - mean).div(torch.sqrt(var +1e-05))
    assert torch.allclose(x_norm, x_ln)
    
def test_pre_norm():
    x = torch.randn(2,3,4)
    fn = nn.Linear(4,1)
    pre_norm = PreNorm(dim=4, fn = fn)
    x_pn = pre_norm(x)
    
    # manual
    x_pn2 = fn(pre_norm.ln(x))
    assert torch.allclose(x_pn, x_pn2)
    
def test_feedforward():
    x = torch.randn(2,3,4)
    ff = FeedForward(dim=4)
    assert ff(x).shape == (2,3,4)

def test_MLP():
    dimensions = [2,3,4]
    x = torch.randn(1,2)
    mlp = MLP(dimensions)
    out = mlp(x)
    assert out.shape == (1, 4)

def test_multi_headed_attention():
    batch_size = 2
    num_cat = 4
    embd_dim= 5
    cat_embd = torch.randn(batch_size, num_cat, embd_dim)
    num_head = 10
    head_dim = 3

    my_attn = Attention(dim=embd_dim, num_head=num_head, dim_head=head_dim)
    
    class AttentionGithub(nn.Module):
        def __init__(
            self,
            dim,
            heads = 8,
            dim_head = 16,
            dropout = 0.):
            super().__init__()
            inner_dim = dim_head * heads
            self.heads = heads
            self.scale = dim_head ** -0.5

            self.to_qkv = my_attn.to_qkv # nn.Linear(dim, inner_dim * 3, bias = False)
            self.to_out = my_attn.to_out # nn.Linear(inner_dim, dim)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            h = self.heads
            q, k, v = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            attn = sim.softmax(dim = -1)
            dropped_attn = self.dropout(attn)

            out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)', h = h)
            return self.to_out(out), attn
        
    att_gh = AttentionGithub(dim=embd_dim,
                          heads=num_head,
                          dim_head=head_dim, 
                          dropout=0)
    gh_out = att_gh(cat_embd)[0]
    my_out = my_attn(cat_embd)
    
    # manual attention implementation
    input_dim = embd_dim
    inner_dim = num_head * head_dim
    to_qkv = my_attn.to_qkv # nn.Linear(input_dim, inner_dim * 3)
    to_out = my_attn.to_out # nn.Linear(inner_dim, input_dim) # residual connection
    attn_dropout = nn.Dropout(0)
    res_dropout = nn.Dropout(0)

    q,k,v = to_qkv(cat_embd).chunk(3, dim=-1) # (B, T, inner_dim *3) -> (B, T, inner_dim) * 3
    q,k,v = map(lambda x: x.reshape(batch_size, num_cat, num_head, head_dim).transpose(1,2), [q,k,v]) # (B, T, inner_dim) * 3 -> (B, H, T, head_dim) * 3
    sim = q @ k.transpose(-1,-2) # (B, H, T, D), (B, H, T, D) -> (B, H, T, T)
    sim = sim * (head_dim ** -0.5)
    sim = sim.softmax(dim=-1) # (B, H, T, T) -> (B, H, T, T)
    sim = attn_dropout(sim)
    attention = sim @ v # (B, H, T, T), (B, H, T, D) -> (B, H, T, D) 

    attention = attention.transpose(1,2).reshape(batch_size, num_cat, -1) # (B, H, T, D) -> (B, T, H * D)
    attention = to_out(attention) # (B, T, H*D) -> (B, T, input_dim)
    attention = res_dropout(attention)
    man_out = attention

    # assert all
    assert torch.allclose(gh_out, my_out)
    assert torch.allclose(gh_out, man_out)
    
    
def test_encoder_layer():
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

    # Transformer Encoder
    class EncoderLayer(nn.Module):
        def __init__(self, 
                    dim, 
                    num_head, 
                    dim_head,
                    dropout=0):
            super().__init__()
            self.ln_1 = nn.LayerNorm(dim)
            self.multi_head_attention = Attention(dim=dim, 
                                                num_head=num_head,
                                                dim_head = dim_head,
                                                dropout=dropout)
            self.ln_2 = nn.LayerNorm(dim)
            self.ff = FeedForward(dim)
            
        def forward(self, x):
            x = x + self.multi_head_attention(self.ln_1(x))
            x = x + self.ff(self.ln_2(x))
            return x
        
    encoder_layer = EncoderLayer(dim=embd_dim,
                                num_head=num_head,
                                dim_head=head_dim)
    out1 = encoder_layer(cat_embd)


    class EncoderLayer2(nn.Module):
        def __init__(self, dim, num_head, dim_head, dropout=0):
            super().__init__()
            attn_og = encoder_layer.multi_head_attention
            ff_og = encoder_layer.ff
            self.attn = Residual(PreNorm(attn_og, dim=dim))
            self.ff = Residual(PreNorm(ff_og, dim=dim))
            # self.attn = Residual(PreNorm(fn=Attention(dim=dim,
            #                                           num_head=num_head,
            #                                           ), 
            #                              dim=dim))
            # self.ff = Residual(PreNorm(fn=FeedForward(dim=dim),
            #                            dim=dim))
            
        def forward(self, x):
            x = self.attn(x)
            x = self.ff(x)
            return x
        
    encoder_layer2 = EncoderLayer2(dim=embd_dim,
                                num_head=num_head,
                                dim_head=head_dim)
    out2 = encoder_layer2(cat_embd)

    assert torch.allclose(out1, out2)
    
    
    
def test_tabtransformer():
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

    tab_transformer = TabTransformer(dim=embd_dim,
                                    num_head=num_head,
                                    dim_head=head_dim,
                                    num_layers=3,
                                    num_cat=num_cat,
                                    num_numerical=num_numerical,
                                    hidden_mults=(3,4),
                                    dropout=0)


    t_out = tab_transformer(x_categorical, x_numerical)
    assert t_out.shape == (batch_size, 1)