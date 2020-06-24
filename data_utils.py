import torchvision.datasets as datasets
import  torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

def dataloader_init(config):
    train_batch_size = config['train_batch_size']
    valid_batch_size = config['valid_batch_size']
    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    training_loader = DataLoader(training_data, 
                             batch_size=train_batch_size, 
                             shuffle=True)

    validation_loader = DataLoader(validation_data,
                               batch_size=valid_batch_size,
                               shuffle=False)

    # data_var = np.var(training_data.data / 255.0)

    return training_loader, validation_loader

