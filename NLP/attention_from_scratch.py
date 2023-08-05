import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
## Generate a batch of data for testing
batch_size = 5
vocab_size = 11
max_length = 10
# simulate a single batch of data
data = torch.from_numpy(np.random.randint(1,vocab_size, size=(batch_size, max_length))) # (5, 10)
src = data

def causal_attention(seq):
    # B: batch_size = 5, T: max length of sequence = 10
    B,T = seq.shape
    
    # (1) Get input embedding
    n_embd = 16 # embedding dimension, this can be any number
    seq_emb = nn.Embedding(vocab_size, n_embd) # Create torch embedding layer
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, C: n_embd = 16
    x = seq_emb(seq) # (B, T, C)
        
    # (2) Apply Linear layer to get query, key, value
    c_attn = nn.Linear(n_embd, 3 * n_embd)
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, 3*C: 3*n_embd = 48
    out = c_attn(x) # (B,T C) -> (B, T, 3*C)
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, C: n_embd = 16
    q, k, v = out.split(n_embd, dim=-1) # (B,T,3*C) -> (B,T,C), (B,T,C), (B,T,C)
        
    # (3) Apply Dot product between query and keys to get the attention
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, T: max length of sequence = 10
    attention = q @ k.transpose(1,2) # (B,T,C), (B,C,T) -> (B,T,T)
    
    # (4) Apply normalize the attention based on the embd_dim
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, T: max length of sequence = 10
    atttention = attention * (1.0 / math.sqrt(k.shape[2])) # (B,T,T)
    
    # (5) Apply causal mask
    mask = torch.tril(torch.ones(1, T,T)) # (1, T ,T)
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, T: max length of sequence = 10
    masked_attention = attention.masked_fill_(mask==0, -np.inf) # (B,T,T) -> (B,T,T)
    
    # (6) Normalise attention with softmax
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, T: max length of sequence = 10
    normalized_masked_attention = F.softmax(masked_attention, dim=-1) # (B,T,T) -> (B,T,T)

    # (7) Apply dropout. called attention dropout because it is applied on the attention
    attn_dropout = nn.Dropout(0.2)
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, T: max length of sequence = 10
    normalized_masked_attention = attn_dropout(normalized_masked_attention) # (B,T,T) -> (B,T,T)

    # (8) Apply attention weight on value key
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, C: n_embd = 16
    v_attention = normalized_masked_attention @ v # (B,T,T), (B,T,C) -> (B,T,C)

    # (9) Apply output projection layer
    c_proj = nn.Linear(n_embd, n_embd)
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, C: n_embd = 16
    y = c_proj(v_attention) # (B,T,C) -> (B,T,C)
    
    # (10) Apply another dropout. Called residual dropout because it is applied on the final remaining layer
    resid_dropout = nn.Dropout(0.2)    
    # Output shape: B: batch_size = 5, T: max length of sequence = 10, C: n_embd = 16
    y = resid_dropout(y) # (B,T,C) -> (B,T,C)
    return y

causal_attention(src).shape