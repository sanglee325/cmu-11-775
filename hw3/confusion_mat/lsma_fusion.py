import os
import random
import csv
import pickle

import torch
import torchvision

import numpy as np
from PIL import Image


class LSMAFusion(torch.utils.data.Dataset):

    def __init__(self, data_path="./data", data_type="train"):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        
        self.mfcc_dir = data_path + "/mfcc/"
        self.cnn_dir = data_path + "/cnn/"
        self.Y_dir = data_path + "/labels/"

        # load label file
        if data_type == "train":
            csvpath = self.Y_dir + "train.csv"
        elif data_type == "val":
            csvpath = self.Y_dir + "val.csv"

        # using a small part of the dataset to debug
        self.id_list, self.id_categories = self.parse_csv(csvpath)
        
        # load video data
        self.id_feat = {}
        for feat_id in self.id_categories.keys():
            # load mfcc
            mfcc_path = self.mfcc_dir + feat_id + ".mfcc.csv"
            try:
                feat = np.genfromtxt(mfcc_path, delimiter=";", dtype="float")
            except:
                feat = np.zeros((40,313))
            mfcc_ft = torch.from_numpy(feat).float().unsqueeze(0)

            # load cnn feature
            feat_path = self.cnn_dir + feat_id + ".pkl"
            with open(feat_path, 'rb') as f:
                _, frame_feature = pickle.load(f)
            cnn_ft = torch.from_numpy(frame_feature)
            self.id_feat[feat_id] = (mfcc_ft, cnn_ft)

        self.length = len(self.id_list)

    @staticmethod
    def parse_csv(filepath):
        id_categories = {}
        id_list = []
        for line in open(filepath).readlines()[1:]:
            feat_id, category = line.strip().split(",")
            id_categories[feat_id] = category
            id_list.append(feat_id)
            
        return id_list, id_categories
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        feat_key = self.id_list[idx]

        (mfcc, cnn) = self.id_feat[feat_key]
        Y = torch.tensor(int(self.id_categories[feat_key]))
        return mfcc, cnn, Y

    def get_idlist(self):
        return self.id_list


class LSMATestFusion(torch.utils.data.Dataset):

    def __init__(self, data_path="./data"):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        
        self.mfcc_dir = data_path + "/mfcc/"
        self.cnn_dir = data_path + "/cnn/"
        self.Y_dir = data_path + "/labels/"

        # load label file
        csvpath = self.Y_dir + "test_for_students.csv"

        self.id_list = self.parse_csv(csvpath)
        
        # load video data
        self.id_feat = {}
        for feat_id in self.id_list:
            # load mfcc
            mfcc_path = self.mfcc_dir + feat_id + ".mfcc.csv"
            try:
                feat = np.genfromtxt(mfcc_path, delimiter=";", dtype="float")
            except:
                feat = np.zeros((40,313))
            mfcc_ft = torch.from_numpy(feat).float().unsqueeze(0)

            # load cnn feature
            feat_path = self.cnn_dir + feat_id + ".pkl"
            with open(feat_path, 'rb') as f:
                _, frame_feature = pickle.load(f)
            cnn_ft = torch.from_numpy(frame_feature)
            self.id_feat[feat_id] = (mfcc_ft, cnn_ft)

        self.length = len(self.id_list)

    @staticmethod
    def parse_csv(filepath):
        id_list = []
        for line in open(filepath).readlines()[1:]:
            feat_id, _ = line.strip().split(",")
            id_list.append(feat_id)
        return id_list
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        feat_key = self.id_list[idx]

        (mfcc, cnn) = self.id_feat[feat_key]
        return mfcc, cnn

    def get_idlist(self):
        return self.id_list
