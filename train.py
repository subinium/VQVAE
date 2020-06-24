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
from utils import *
from logger import *
from validate import validate


def train(args, config):
    # random seed
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('cuda is available')

    # Model
    model = VQVAE(config, device).to(device)
    print('Number of Parameters: ', sum(param.numel() for param in model.parameters()))

    # Logger
    logger = prepare_logger(args.output_dir, args.summary_dir, args.checkpoint_dir, args.log_dir)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'], amsgrad=False)

    # Dataset
    train_loader, valid_loader = dataloader_init(config)

    epoch_offset = 0
    iteration = 0
    if args.load_checkpoint :
        model, optimizer, iteration = load_model(model, optimizer, iteration, args.load_checkpoint)
        print(f'Load chcekpoint \"{args.load_checkpoint}\" complete.')
        epoch_offset = iteration // len(train_loader)

    # Training
    model.train()
    
    for epoch in range(epoch_offset, config['num_epochs']):
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
            training_loss_dict["training/perplexity"] = perplexity.item()
            training_loss_dict["training/loss"] = loss.item()

            logger.add_scalars(training_loss_dict, iteration)

            print(f'Epoch {epoch} : {idx}/{len(train_loader)} : {loss.item()}\r', end='')

            if iteration % config['iters_per_validate'] == 0:
                validate(model, valid_loader, iteration, logger)

                checkpoint_path = os.path.join(args.output_dir, args.checkpoint_dir, f'checkpoint_{iteration}')
                save_model(model, optimizer, iteration, checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='result', help='directory to save checkpoints')
    parser.add_argument('-s', '--summary_dir', type=str, default='summary', help='directory to save tensorboard logs')
    parser.add_argument('-m', '--checkpoint_dir', type=str, default='model', help='directory to save checkpoints')
    parser.add_argument('-l', '--log_dir', type=str, default='log', help='directory to save checkpoints')
    parser.add_argument('-c', '--load_checkpoint', type=str, default=None, help='load checkpoint path')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)

    train(args, config)
