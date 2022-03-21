import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os

from lsma_fusion import LSMAFusion, LSMATestFusion


DATA_DIR = "../data"

def load_dataset():

    val_dataset = LSMAFusion(DATA_DIR, data_type="val")

    return val_dataset

def load_dataloader(val_dataset, batch_size):

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4)

    return val_loader

