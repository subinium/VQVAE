import os
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F

from model.layer import *

class VectorQuantizer(nn.Module):
    def __init__(self, config, device):
        self.device = device 
        self.embedding_dim = config['embedding_dim']
        self.num_embeddings = config['num_embeddings']
        self.commitment_cost = config['beta']
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.num_embedding, 1/self.num_ebedding) # why uniform init -1, 1?

    def forward(self, input):
        # BCHW -> BHWC : because codebook find most closest *channel set*
        input = input.permute(0, 2, 3, 1).contiguous()
        input_shape = input.shape
        
        # Flatten input
        flat_input = input.view(-1, self.embedding_dim)
        
        # Calculate distances
        # ||z_e - z_q||_2 = z_e**2 + z_q**2 - 2*z_e*z_q
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Find closest codebook components
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) 
        
        # OneHot Masking for gradient: You can use torch.eye for one-hot encoding
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1) 
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape) # reshape
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), input) # z_e loss : encoder components
        q_latent_loss = F.mse_loss(quantized, input.detach()) # z_q loss : codebook components
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # This is a technique to prevent the gradient of the decoder model 
        # to be used later from updating the codebook.
        quantized = input + (quantized - input).detach()  
        
        # Perplexity : Information Theory (TBD)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

# Encoder & Decoder : Based on ResNet
class Encoder(nn.Module):
    def __init__(self, config):
        in_channels = config['in_channels']
        num_hiddens = config['num_hiddens']
        self.model = nn.Sequential(
            Conv2dInit(in_channels, num_hiddens//2, 4, 2, 1),
            nn.ReLU(True),
            Conv2dInit(num_hiddens//2, num_hiddens, 4, 2, 1),
            nn.ReLU(True),
            Conv2dInit(num_hiddens, num_hiddens, 3, 1, 1), 
            ResidualBlocks(num_hiddens, num_hiddens,
                            num_residual_layers=config['num_residual_layers'],
                            num_residual_hiddens=config['num_residual_hiddens']))

    def forward(self, input):
        return self.model(input)

class Decoder(nn.Module):
    def __init__(self, config):
        in_channels = config['in_channels']
        num_hiddens = config['num_hiddens']
        self.model = nn.Sequential(
            Conv2dInit(in_channels, num_hiddens, 3, 1, 1),
            ResidualBlocks(num_hiddens, num_hiddens,
                            num_residual_layers=config['num_residual_layers'],
                            num_residual_hiddens=config['num_residual_hiddens']),
            nn.ConvTransposed2d(num_hiddens, num_hiddens//2, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTransposed2d(num_hiddens, 3, 4, 2, 1),
        )

    def forward(self, input):
        return self.model(input)