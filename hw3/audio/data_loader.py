import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os

from lsma_audio import LSMAAudio, LSMATestAudio
from config import *


DATA_DIR = "../data"

def load_dataset():
    audio_preprocess = transforms.Compose([
                        transforms.Normalize(mean=[0.485], std=[0.229]),
                        ])

    train_dataset = LSMAAudio(DATA_DIR, data_type="train", transforms=audio_preprocess)
    val_dataset = LSMAAudio(DATA_DIR, data_type="val", transforms=audio_preprocess)
    test_dataset = LSMATestAudio(DATA_DIR, transforms=audio_preprocess)

    return train_dataset, val_dataset, test_dataset

def load_dataloader(train_dataset, val_dataset, test_dataset, batch_size):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=ARGS.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=ARGS.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=ARGS.num_workers)

    return train_loader, val_loader, test_loader

