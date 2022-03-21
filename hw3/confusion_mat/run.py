import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import os.path as osp

from tqdm import tqdm
import numpy as np

from data_loader import load_dataset, load_dataloader
from confmat import plot_confusion_matrix
import models.resnet2d as resnet2d
import models.mlp as mlp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_embedding(audio, video, mfcc, cnn):
    audio.eval()
    video.eval()

    with torch.no_grad():
        audio_x = audio(mfcc, feat=True)
        video_x = video(cnn, feat=True)

    x = torch.cat((audio_x, video_x), dim=1)
    
    return x

def compute_confmat(val_loader, audio, video, fusion):
    audio_mat = torch.zeros(num_classes, num_classes)
    video_mat = torch.zeros(num_classes, num_classes)
    fusion_mat = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for i, (mfcc, cnn, classes) in enumerate(val_loader):
            mfcc = mfcc.to(device)
            outputs = audio(mfcc)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                audio_mat[t.long(), p.long()] += 1

    with torch.no_grad():
        for i, (mfcc, cnn, classes) in enumerate(val_loader):
            cnn = cnn.to(device)
            outputs = video(cnn)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                video_mat[t.long(), p.long()] += 1

    with torch.no_grad():
        for i, (mfcc, cnn, classes) in enumerate(val_loader):
            mfcc = mfcc.to(device)
            cnn = cnn.to(device)
            
            x = generate_embedding(audio, video, mfcc, cnn)
            outputs = fusion(x)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                fusion_mat[t.long(), p.long()] += 1

    return [audio_mat, video_mat, fusion_mat]

if __name__ == '__main__':# load model
    num_classes = 15
    input_size = 640

    model = mlp.MLPNetwork2(input_size=input_size, num_classes=num_classes)
    
    model_video = mlp.MLPNetwork(input_size=512, num_classes=num_classes)

    model_audio = resnet2d.resnet34()
    model_audio.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)                    
    num_features = model_audio.fc.in_features
    model_audio.fc = nn.Linear(num_features, num_classes)
    
    model.to(device)
    model_audio.to(device)
    model_video.to(device)

    lr=1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_audio = optim.Adam(model_audio.parameters(), lr=lr)
    optimizer_video = optim.Adam(model_video.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss().to(device)

    scaler = torch.cuda.amp.GradScaler()

    # resume model weight
    audio_path = "./ckpt/audio_resnet34.pth"
    audio_ckpt = torch.load(audio_path)
    model_audio.load_state_dict(audio_ckpt['net'])
    optimizer_audio.load_state_dict(audio_ckpt['optimizer'])

    video_path = "./ckpt/video_mlp.pth"
    video_ckpt = torch.load(video_path)
    model_video.load_state_dict(video_ckpt['net'])
    optimizer_video.load_state_dict(video_ckpt['optimizer'])
    
    fusion_path = "./ckpt/fusion.pth"
    fusion_ckpt = torch.load(fusion_path)
    model.load_state_dict(fusion_ckpt['net'])
    optimizer.load_state_dict(fusion_ckpt['optimizer'])

    # load dataset
    val_dataset = load_dataset()
    val_loader = load_dataloader(val_dataset, 256)

    confmat_list = compute_confmat(val_loader, model_audio, model_video, model)

    result_dir = 'result'
    audio_path = osp.join(result_dir, 'audio.png')
    video_path = osp.join(result_dir, 'video.png')
    fusion_path = osp.join(result_dir, 'fusion.png')
    conf_path = [audio_path, video_path, fusion_path]

    labels = [i for i in range(0,15)]  
    for i, (confmat) in enumerate(confmat_list):
        data = confmat.tolist()
        plot_confusion_matrix(data, labels, conf_path[i])


