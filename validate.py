from collections import defaultdict

import torch 
import torch.nn as nn 
import torch.nn.functional as F


def validate(model, valid_loader, iteration, logger):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    validation_loss_dict = defaultdict(lambda x : 0)
    for idx, (image, label) in enumerate(valid_loader):
        image = image.to(device)
        vq_loss, image_recon, perplexity = model(image)

        # Loss
        recon_loss = F.mse_loss(image_recon, image)
        loss = recon_loss + vq_loss
        
        # Logger
        validation_loss_dict["validation/recon_loss"] = recon_loss.item()
        validation_loss_dict["validation/vq_loss"] = vq_loss.item()
        validation_loss_dict["validation/loss"] = loss.item()

        logger.add_scalars(validation_loss_dict, iteration)
    
    model.train()