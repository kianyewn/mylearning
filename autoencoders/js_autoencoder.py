import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    # source: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
            (Eg if you use gaussian noise, in layer 2, gradient will not flow from layer 2 to layer 1 for backpropagation)
    """
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        # Buffers are not learnable and are used for tensors that should be included in the state of the module
        # The self.noise buffer is used to store a tensor of zeros, and it remains constant during training.
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        """
        if self.is_relative_true == True, this means that gradient will not flow from 'X' to 'noise'
        """
        if self.training and self.sigma != 0:
            # compute the scale of the noise
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, dimensions=[], dropout=0.1):
        super().__init__()
        self.dimensions = dimensions
        self.dropout=dropout
        
        self.layers = nn.ModuleList()
        dropout = nn.Dropout(dropout)
        
        for i, (ft_in, ft_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            self.layers.append(nn.Linear(ft_in, ft_out))
#             if i != len(hidden_dimensions)-2:
            self.layers.append(nn.BatchNorm1d(ft_out))
            self.layers.append(nn.SiLU())
            self.layers.append(dropout)
                
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out
    
X = torch.randn(2,10)
encoder = Encoder(dimensions=[10,50,20])
encoder_out = encoder(X)
            
class Decoder(nn.Module):
    def __init__(self, dimensions=[], dropout=0.1):
        super().__init__()
        self.dimensions = dimensions
        self.dropout=dropout
        
        self.layers = nn.ModuleList()
        dropout = nn.Dropout(dropout)
        
        for i, (ft_in, ft_out) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            self.layers.append(nn.Linear(ft_in, ft_out))
            if i != len(dimensions)-2:
                self.layers.append(nn.BatchNorm1d(ft_out))
                self.layers.append(nn.SiLU())
                self.layers.append(dropout)
                
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out
    
num_labels = 5
decoder = Decoder(dimensions=[20,40,10])
decoder_out = decoder(encoder_out)
        

class MLP(nn.Module):
    def __init__(self, dimensions, dropout=0):
        super().__init__()
        self.dimensions = dimensions
        self.layers = nn.ModuleList()
        for i, (in_ft, out_ft) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            self.layers.append(nn.Linear(in_ft, out_ft))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.Dropout(dropout))
            
    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out
            
class AutoEncoder(nn.Module):
    def __init__(self, 
                 num_labels,
                 num_features, 
                 encoder_dimensions=[], 
                 decoder_dimensions=[],
                 mlp_dimensions=[],
                 encoder_dropout=0,
                 decoder_dropout=0,
                 mlp_dropout=0,
                 sigma=0.1,
                 is_relative_detach=True):
        
        super().__init__()
        
        self.gaussian_noise = GaussianNoise(sigma=sigma, is_relative_detach=is_relative_detach)
        self.input_bn = nn.BatchNorm1d(num_features)
        
        self.encoder = Encoder(dimensions=encoder_dimensions,
                          dropout=encoder_dropout)
        
        self.decoder = Decoder(dimensions=decoder_dimensions,
                              dropout=decoder_dropout)
        
        mlp_dimensions = [num_features + encoder_dimensions[-1]] + mlp_dimensions
        
        self.mlp = MLP(dimensions=mlp_dimensions,
                       dropout=mlp_dropout)
        
        self.decoder_projection_head = nn.Linear(decoder_dimensions[-1], num_labels)
        self.mlp_projection_head = nn.Linear(mlp_dimensions[-1], num_labels)
        
        
    def forward(self, X, y=None):
        """ X: (B,N), y: (B, num_labels)
        """
        # normalize and add noise
        X = self.input_bn(X)
        X = self.gaussian_noise(X)
        
        encoder_out = self.encoder(X) # (B, n*)
        decoder_out = self.decoder(encoder_out) # (B, n)
        
        mlp_in = torch.cat([X, encoder_out], dim=-1) # (B, n + n*)
        mlp_out = self.mlp(mlp_in)
        
        recon_loss = None
        ae_out_loss = None
        mlp_out_loss = None
        if y is not None:
            # Calculate feature reconstruction loss
            recon_loss = nn.functional.mse_loss(decoder_out, X)

            # Calculate decoder cross entropy loss for multi-label classification
            ae_out = self.decoder_projection_head(decoder_out)
            ae_out_loss = torch.nn.functional.binary_cross_entropy_with_logits(ae_out, y)

            # Calculate mlp cross entropy loss for multi-label classification
            mlp_out = self.mlp_projection_head(mlp_out)
            mlp_out_loss =  torch.nn.functional.binary_cross_entropy_with_logits(mlp_out, y)
        
        return X, (recon_loss, ae_out_loss, mlp_out_loss)
    
X = torch.randn(10,20)
y = torch.where(torch.zeros(10,10).float().uniform_() < 0.5, 1,0).type(torch.float)
    
ae = AutoEncoder(
    num_features= 20,
    num_labels = 10,
    encoder_dimensions = [20,30,10],
    decoder_dimensions = [10, 30, 20],
    mlp_dimensions=[20,30])
        

out, losses = ae(X, y)