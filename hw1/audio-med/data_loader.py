import sys, os
import torch
import librosa
import numpy as np
import pandas as pd
from torch import Tensor
from scipy.io import wavfile
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# /home/psl/hdd/cmu/data/hw1/mfcc
class Freesound(Dataset):
    def __init__(self, data_root, transform=None, mode="train"):
        # setting directories for data
        self.data_root = data_root
        self.mode = mode
        if self.mode is "train":
            self.data_dir = os.path.join(data_root, "audio_train")
            self.csv_file = pd.read_csv(os.path.join(data_root, "train.csv"))
        elif self.mode is "test":
            self.data_dir = os.path.join(data_root, "audio_test")
            self.csv_file = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))

        # dict for mapping class names into indices. can be obtained by 
        # {cls_name:i for i, cls_name in enumerate(csv_file["label"].unique())}
        self.classes = pd.read_csv(os.path.join(data_root, "labels", "cls_map.csv")

        self.transform = transform
        
    def __len__(self):
        return self.csv_file.shape[0] 

    def __getitem__(self, idx):
        filename = self.csv_file["fname"][idx]
        
        rate, data = wavfile.read(os.path.join(self.data_dir, filename))

        if self.transform is not None:
            data = self.transform(data)

        if self.mode is "train":
            label = self.classes[self.csv_file["label"][idx]]
            return data, label

        elif self.mode is "test":
            return data
