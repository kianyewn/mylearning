## Note copied the data from Andrej Karparthy's NanoGPT
"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
# input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
input_file_path = os.path.join(os.getcwd(), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
# train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

train_ids.tofile(os.path.join(os.getcwd(), 'train.bin'))
val_ids.tofile(os.path.join(os.getcwd(), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
# with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
#     pickle.dump(meta, f)
    
with open(os.path.join(os.getcwd(), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import reduce

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = ['pad','mask'] + sorted(list(set(text)))
vocab_size = len(chars) + 2

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# get batch data instead of torch data loader
def get_batches(data):
    sample_indices = torch.randint(len(data)-block_size, (batch_size,))
    # x =  [data[i:i+block_size] for i in range(len(data)-block_size)]
    x =  [data[i:i+block_size] for i in sample_indices]
    # y = [data[i+1:i+block_size+1] for i in range(len(data)-block_size)]
    y = [data[i+1:i+block_size+1] for i in sample_indices]
    
    bX, by = torch.stack(x), torch.stack(y)
    return bX, by

# remember to send your batches to the device that you want
# bX, by = get_batches(train_data)

class MLM(nn.Module):
    def __init__(self, 
                 model,
                 mask_prob=0.15,
                 replace_prob=0.9,
                 random_prob=0.2,
                 pad_token_id = 0,
                 mask_token_id = 1,
                 ignore_token_ids =[],
                 num_tokens=None):
        super().__init__()
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.ignore_token_ids = ignore_token_ids
        
        self.num_tokens = num_tokens
        self.model = model
        
    def masking_with_ignore_tokens(self, tensor, token_ids):
        init_no_mask = torch.full_like(tensor, fill_value=False, dtype=bool)
        return reduce(lambda acc, el: acc | tensor==el, token_ids, init_no_mask)
    
    def forward(self, X, y):
        # boolean matrix of tokens that should be masked according to mask prob
        all_mask = torch.zeros_like(X).float().uniform_(0,1) < self.mask_prob # (B, T)
        # boolean matrix of tokens that should not be masked because they are in ignore_token_ids
        all_no_mask = self.masking_with_ignore_tokens(X, self.ignore_token_ids) # (B, T)
        # boolean matrix of tokens that should be masked according to mask_prob and are not in ignore_token_ids
        all_mask &= ~all_no_mask # (B, T)
        
        # create the batch labels for y.
        # replace tokens that are not masked with pad token id. 
        # Used to prevent calculating loss function on tokens not selected for mlm
        y = y.masked_fill_(~all_mask, self.pad_token_id) # (B, T)
        
        # Apply random tokens on positions that should be random according to random_prob
        random_mask = torch.zeros_like(X).float().uniform_(0,1) < self.random_prob # (B,T)
        random_no_mask = self.masking_with_ignore_tokens(X, self.ignore_token_ids) # (B, T)
        random_mask &= ~random_no_mask
        
        random_indices = torch.nonzero(random_mask, as_tuple=True) # (pos x,), (pox,y)
        random_tokens = torch.randint(self.num_tokens, size=(len(random_indices[0]),)) # (num_random_tokens)
        
        masked_X = X.detach().clone() # (B, T)
        masked_X[random_indices] = random_tokens # (B, T)
        
        # Apply mask tokens on positions that should be mask according to replace_prob (which is for mask tokens)
        # out of the remaining 10%, 'random_prob' tokens are random_tokens, the remainder are unchanged tokens 
        masking_mask = torch.zeros_like(X).float().uniform_(0,1) < self.replace_prob
        
        # Dont need the line below because if we multiply by random_mask, resulting mask of those ignore_tokens will be False
        # masking_no_mask = self.masking_with_ignore_tokens(X, self.ingore_token_ids)
        
        # overwrite the tokens that should be random with mask token. 90% tokens of all masks are mask tokens
        masking_mask = masking_mask * random_mask  # (B, T) 
        masked_X = masked_X.masked_fill_(masking_mask, self.mask_token_id) # (B,T)
        
        model_out = self.model(masked_X)
        
        # need to transpose because it is multi-dimensional loss, pytorch requires input logit to be (B, E, T)
        mlm_loss = torch.nn.functional.cross_entropy(model_out.transpose(-1,-2), y)
        return mlm_loss
        
        
# mlm = MLM(model, num_tokens=vocab_size)
# # mlm.masking_with_ignore_token(bX, token_ids=[])
# mlm(bX, by)

class Model(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.linear = nn.Linear(n_embd, vocab_size)

    def forward(self,X):
        emb = self.emb(X)
        logits = self.linear(emb)
        return logits

model = Model(vocab_size, n_embd)
mlm = MLM(model, num_tokens=vocab_size)
optimizer = torch.optim.AdamW(mlm.parameters(), lr=1e-3)

num_eps = 10 
for n in range(num_eps):
    # remember to send your batches to the device that you want
    bX, by = get_batches(train_data)
    
    mlm_loss = mlm(bX, by)
    optimizer.zero_grad()
    mlm_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        # remember to send your batches to the device that you want
        bX2, by2 = get_batches(val_data)
        mlm_loss_val = mlm(bX2, by2)
    print(mlm_loss, mlm_loss_val)
    
# tensor(4.4087, grad_fn=<NllLoss2DBackward0>) tensor(4.2252)
# tensor(4.2232, grad_fn=<NllLoss2DBackward0>) tensor(4.0361)
# tensor(4.0294, grad_fn=<NllLoss2DBackward0>) tensor(3.8470)
# tensor(3.8370, grad_fn=<NllLoss2DBackward0>) tensor(3.7045)
# tensor(3.6711, grad_fn=<NllLoss2DBackward0>) tensor(3.5084)
# tensor(3.5155, grad_fn=<NllLoss2DBackward0>) tensor(3.3901)
# tensor(3.3710, grad_fn=<NllLoss2DBackward0>) tensor(3.2596)
# tensor(3.2521, grad_fn=<NllLoss2DBackward0>) tensor(3.1300)
# tensor(3.1245, grad_fn=<NllLoss2DBackward0>) tensor(3.0468)
# tensor(3.0428, grad_fn=<NllLoss2DBackward0>) tensor(2.9455)