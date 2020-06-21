import os
import json
import random
import argparse
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import make_grid

from model.model import *
from data_utils import *
from logger import *
from validate import validate


def train(config):
    # random seed
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('cuda is available')

    # Model
    model = VQVAE(config, device).to(device)
    print('Number of Parameters: ', sum(param.numel()
                                        for param in model.parameters()))

    # Logger
    logger = prepare_logger(
        config['output_dir'], config['summary_dir'], config['checkpoint_dir'], config['log_dir'])

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'], amsgrad=False)

    # Dataset
    train_loader, valid_loader = dataloader_init(config)

    # Training
    model.train()
    iteration = 0
    for epoch in range(config['num_epochs']):
        training_loss_dict = defaultdict(lambda: 0)
        for idx, (image, label) in enumerate(train_loader):
            iteration += 1

            image = image.to(device)
            vq_loss, image_recon, perplexity = model(image)

            # Loss
            recon_loss = F.mse_loss(image_recon, image)
            loss = recon_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logger
            training_loss_dict["training/recon_loss"] = recon_loss.item()
            training_loss_dict["training/vq_loss"] = vq_loss.item()
            training_loss_dict["training/loss"] = loss.item()

            logger.add_scalars(training_loss_dict, iteration)

            print(
                f'Epoch {epoch} : {idx}/{len(train_loader)} : {loss.item()}\r', end='')

            if iteration % config['iters_per_validate'] == 0:
                validate(model, valid_loader, iteration, logger)


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)

    train(config)
