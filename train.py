import os
import json

import random
import numpy as np
import argparse

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import Adam

from torchvision.utils import make_grid
from model.model import *

from data_utils import *

def train(config):
    torch.manual_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VQVAE(config, device).to(device)
    optimizer = Adam(model.parameters(), lr=config['lr'], amsgrad=False)

    train_loader, valid_loader, data_var = dataloader_init(config)
    # Training
    model.train()
    
    for epoch in range(config['num_epochs']):
        for idx, (image, label) in enumerate(train_loader):
            image = image.to(device)
            vq_loss, image_recon, perplexity = model(image)
            recon_loss = F.mse_loss(image_recon, image) / data_var
            loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
    
    train(config)