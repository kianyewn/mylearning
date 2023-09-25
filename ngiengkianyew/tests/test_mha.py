import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum
from sequences.tabtransformer import Attention

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
