import os
import random
import csv

import torch
import torchvision
from torchvision import transforms

import numpy as np
from PIL import Image


class LSMAAudio(torch.utils.data.Dataset):

    def __init__(self, data_path="./data", data_type="train", transforms=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        
        self.X_dir = data_path + "/mfcc/"
        self.Y_dir = data_path + "/labels/"
        self.transforms = transforms

        # load label file
        if data_type == "train":
            csvpath = self.Y_dir + "train.csv"
        elif data_type == "val":
            csvpath = self.Y_dir + "val.csv"

        self.id_list, self.id_categories = self.parse_csv(csvpath)
        
        # load mfcc data
        self.id_mfcc = {}
        for mfcc_id in self.id_categories.keys():
            mfcc_path = self.X_dir + mfcc_id + ".mfcc.csv"
            try:
                feat = np.genfromtxt(mfcc_path, delimiter=";", dtype="float")
            except:
                feat = np.zeros((40,313))
            feat_tensor = torch.from_numpy(feat).float().unsqueeze(0)
            self.id_mfcc[mfcc_id] = self.transforms(feat_tensor)

        assert(len(self.id_categories) == len(self.id_mfcc))
        self.length = len(self.id_mfcc)
      
    @staticmethod
    def parse_csv(filepath):
        id_categories = {}
        id_list = []
        for line in open(filepath).readlines()[1:]:
            mfcc_id, category = line.strip().split(",")
            id_categories[mfcc_id] = category
            id_list.append(mfcc_id)
        return id_list, id_categories
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        mfcc_key = self.id_list[idx]

        X = self.id_mfcc[mfcc_key]
        Y = torch.tensor(int(self.id_categories[mfcc_key]))
        return X, Y

    def get_idlist(self):
        return self.id_list

class LSMATestAudio(torch.utils.data.Dataset):

    def __init__(self, data_path="./data", transforms=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        
        self.X_dir = data_path + "/mfcc/"
        self.Y_dir = data_path + "/labels/"
        self.transforms = transforms

        # load label file
        csvpath = self.Y_dir + "test_for_students.csv"

        self.id_list = self.parse_csv(csvpath)
        
        # load mfcc data
        self.id_mfcc = {}
        for mfcc_id in self.id_list:
            mfcc_path = self.X_dir + mfcc_id + ".mfcc.csv"
            try:
                feat = np.genfromtxt(mfcc_path, delimiter=";", dtype="float")
            except:
                feat = np.zeros((40,313))
            feat_tensor = torch.from_numpy(feat).float().unsqueeze(0)
            self.id_mfcc[mfcc_id] = self.transforms(feat_tensor)

        self.length = len(self.id_mfcc)
      
    @staticmethod
    def parse_csv(filepath):
        id_list = []
        for line in open(filepath).readlines()[1:]:
            mfcc_id, _ = line.strip().split(",")
            id_list.append(mfcc_id)
        return id_list
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        mfcc_key = self.id_list[idx]

        X = self.id_mfcc[mfcc_key]
        return X

    def get_idlist(self):
        return self.id_list

if __name__ == '__main__':

    audio_preprocess = transforms.Compose([
                        transforms.Normalize(mean=[0.485], 
                                             std=[0.229]),
                        ])
    
    test_dataset = LSMATestAudio(data_path="../data", transforms=audio_preprocess)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False,
                            num_workers=4)
    
    for i, (x) in enumerate(test_loader):
        x = x
    