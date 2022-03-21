import os
import random
import csv
from select import select

import torch
import torchvision
from torchvision import transforms

import numpy as np
from PIL import Image

class LSMAAudio(torch.utils.data.Dataset):

    def __init__(self, data_path="./data", data_type="train", shuffle=True, transforms=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        
        self.X_dir = data_path + "/mfcc/"
        self.Y_dir = data_path + "/labels/"
        self.transforms = transforms

        # load label file
        if data_type == "train":
            csvpath = self.Y_dir + "train.csv"
        elif data_type == "val":
            csvpath = self.Y_dir + "val.csv"
        elif data_type == "test":
            csvpath = self.Y_dir + "test_for_students.csv"

        # using a small part of the dataset to debug
        self.id_list, self.id_categories = self.parse_csv(csvpath)
        
        # load mfcc data
        self.id_mfcc = {}
        for mfcc_id in self.id_categories.keys():
            mfcc_path = self.X_dir + mfcc_id + ".mfcc.csv"
            try:
                feat = np.genfromtxt(mfcc_path, delimiter=";", dtype="float")
            except:
                feat = np.zeros((40,313))
            feat_tensor = torch.from_numpy(feat).unsqueeze(0)
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
        Y = self.id_categories[mfcc_key]
        return X, Y


class LSMAVideo(torch.utils.data.Dataset):

    def __init__(self, data_path="./data", data_type="train", shuffle=True, transforms=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        
        self.X_dir = data_path + "/videos/"
        self.Y_dir = data_path + "/labels/"
        self.transforms = transforms

        # load label file
        if data_type == "train":
            csvpath = self.Y_dir + "train.csv"
        elif data_type == "val":
            csvpath = self.Y_dir + "val.csv"
        elif data_type == "test":
            csvpath = self.Y_dir + "test_for_students.csv"

        # using a small part of the dataset to debug
        self.id_categories = self.parse_csv(csvpath)
        
        # load video data
        stream = "video"
        self.id_video = {}
        self.id_list = []
        for video_id in self.id_categories.keys():
            video_path = self.X_dir + video_id + ".mp4"
            frames, audio, meta = torchvision.io.read_video(video_path)
            frames = self.select_frames(frames, meta['video_fps'])
            frames = self.preprocess_frames(frames)
            for i, frame in enumerate(frames):
                new_id = video_id + '_' + str(i)
                self.id_video[new_id] = frame

        self.length = len(self.id_list)

    
    def select_frames(self, frames, fps):
        T, H, W, C = frames.shape
        selected_idx = np.arange(0,T,int(fps))
        selected_frames = frames[selected_idx]

        selected_frames = selected_frames / 255
        return selected_frames
    
    def preprocess_frames(self, frames):
        T, H, W, C = frames.shape
        frame_list = []

        for i in range(T):
            frame = frames[i]
            frame = frame.permute(2,0,1)
            frame = self.transforms(frame)
            frame_list.append(frame)

        return frame_list

    @staticmethod
    def parse_csv(filepath):
        id_categories = {}
        
        for line in open(filepath).readlines()[1:]:
            video_id, category = line.strip().split(",")
            id_categories[video_id] = category
            
        return id_categories
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        video_key = self.id_list[idx]
        orig_key = video_key.split('_')[0]

        X = self.id_video[video_key]
        Y = self.id_categories[orig_key]
        return X, Y


if __name__ == '__main__':
    '''
    audio_preprocess = transforms.Compose([
                        transforms.Normalize(mean=[0.485], 
                                             std=[0.229]),
                        ])
    audio = LSMAAudio(transforms=audio_preprocess)
    '''
    
    video_preprocess = transforms.Compose([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225]),
                        ])
    video = LSMAVideo(transforms=video_preprocess)