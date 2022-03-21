import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os

from lsma_fusion import LSMAFusion, LSMATestFusion
from config import *


DATA_DIR = "../data"

def load_dataset():

    train_dataset = LSMAFusion(DATA_DIR, data_type="train")
    val_dataset = LSMAFusion(DATA_DIR, data_type="val")
    test_dataset = LSMATestFusion(DATA_DIR)

    return train_dataset, val_dataset, test_dataset

def load_dataloader(train_dataset, val_dataset, test_dataset, batch_size):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=ARGS.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=ARGS.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=ARGS.num_workers)

    return train_loader, val_loader, test_loader

