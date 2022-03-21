import os
import random
import csv
import pickle

import torch
import torchvision

import numpy as np
from PIL import Image


class LSMAVideoCNN(torch.utils.data.Dataset):

    def __init__(self, data_path="./data", data_type="train"):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        
        self.X_dir = data_path + "/cnn/"
        self.Y_dir = data_path + "/labels/"

        # load label file
        if data_type == "train":
            csvpath = self.Y_dir + "train.csv"
        elif data_type == "val":
            csvpath = self.Y_dir + "val.csv"

        # using a small part of the dataset to debug
        self.id_list, self.id_categories = self.parse_csv(csvpath)
        
        # load video data
        self.id_video = {}
        for video_id in self.id_categories.keys():
            feat_path = self.X_dir + video_id + ".pkl"
            with open(feat_path, 'rb') as f:
                _, frame_feature = pickle.load(f)
            self.id_video[video_id] = torch.from_numpy(frame_feature).unsqueeze(0)

        self.length = len(self.id_list)

    @staticmethod
    def parse_csv(filepath):
        id_categories = {}
        id_list = []
        for line in open(filepath).readlines()[1:]:
            video_id, category = line.strip().split(",")
            id_categories[video_id] = category
            id_list.append(video_id)
            
        return id_list, id_categories
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        video_key = self.id_list[idx]

        X = self.id_video[video_key]
        Y = torch.tensor(int(self.id_categories[video_key]))
        return X, Y

    def get_idlist(self):
        return self.id_list


class LSMATestVideoCNN(torch.utils.data.Dataset):

    def __init__(self, data_path="./data"):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        
        self.X_dir = data_path + "/cnn/"
        self.Y_dir = data_path + "/labels/"

        # load label file
        csvpath = self.Y_dir + "test_for_students.csv"

        self.id_list = self.parse_csv(csvpath)
        
        # load video data
        self.id_video = {}
        for video_id in self.id_list:
            feat_path = self.X_dir + video_id + ".pkl"
            with open(feat_path, 'rb') as f:
                _, frame_feature = pickle.load(f)
            self.id_video[video_id] = torch.from_numpy(frame_feature).unsqueeze(0)

        self.length = len(self.id_list)

    @staticmethod
    def parse_csv(filepath):
        id_list = []
        for line in open(filepath).readlines()[1:]:
            video_id, _ = line.strip().split(",")
            id_list.append(video_id)
        return id_list
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        video_key = self.id_list[idx]

        X = self.id_video[video_key]

        return X
        
    def get_idlist(self):
        return self.id_list
