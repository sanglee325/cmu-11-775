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
import models.resnet2d as resnet2d
import models.mlp as mlp
from config import *

def generate_embedding(audio, video, mfcc, cnn):
    audio.eval()
    video.eval()

    with torch.no_grad():
        audio_x = audio(mfcc, feat=True)
        video_x = video(cnn, feat=True)

    x = torch.cat((audio_x, video_x), dim=1)
    
    return x


def train(model, model_audio, model_video, train_loader, optimizer, scheduler, criterion, scaler, batch_size):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    model.train()

    num_correct = 0
    total_loss = 0
    total = 0

    for i, (mfcc, cnn, y) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        mfcc = mfcc.to(device)
        cnn = cnn.to(device)
        y = y.to(device)
        
        x = generate_embedding(model_audio, model_video, mfcc, cnn)

        outputs = model(x)
        loss = criterion(outputs, y)

        # Update # correct & loss as we go
        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        total += len(x)
        total_loss += float(loss)
        
        # Another couple things you need for FP16. 
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.

        train_acc = 100 * num_correct / total
        train_loss = float(total_loss / total)
        lr_rate = float(optimizer.param_groups[0]['lr'])

        
    print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
                                        epoch + 1, ARGS.epochs, train_acc, train_loss, lr_rate))
        
    return train_acc, train_loss, lr_rate

def validate(model, model_audio, model_video, val_loader, batch_size):
    model.eval()

    num_correct = 0
    total = 0
    for i, (mfcc, cnn, y) in enumerate(tqdm(val_loader)):
        optimizer.zero_grad()

        mfcc = mfcc.to(device)
        cnn = cnn.to(device)
        y = y.to(device)
        
        x = generate_embedding(model_audio, model_video, mfcc, cnn)

        with torch.no_grad():
            outputs = model(x)

        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        total += len(x)
        

    val_acc = 100 * num_correct / total
    print("Validation: {:.04f}%".format(100 * num_correct / total))

    return val_acc

def test(model, model_audio, model_video, test_loader, test_idlist, logdir, name):
    model.eval()

    res = []
    total = 0
    
    for i, (mfcc, cnn) in enumerate(tqdm(test_loader)):
        optimizer.zero_grad()

        mfcc = mfcc.to(device)
        cnn = cnn.to(device)
        
        x = generate_embedding(model_audio, model_video, mfcc, cnn)

        with torch.no_grad():
            outputs = model(x)
        total += len(x)
        pred = torch.argmax(outputs, axis=1)
        res += pred

    log_result = logdir + '/result_' + name +'.csv'
    with open(log_result, "w+") as f:
        f.write("Id,Category\n")
        for i in range(len(test_idlist)):
            f.write("{},{}\n".format(test_idlist[i], res[i]))

    
if __name__ == '__main__':
    # set options for file to run
    logpath = ARGS.log_path
    logfile_base = f"{ARGS.name}_{ARCH}_{ARGS.loss_type}_{ARGS.optim}_S{SEED}_B{BATCH_SIZE}_LR{LR}_E{EPOCHS}"
    logdir = logpath + logfile_base

    set_logpath(logpath, logfile_base)
    print('save path: ', logdir)

    # load model
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

    num_trainable_parameters = 0
    for p in model_audio.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Audio Params: {}".format(num_trainable_parameters))
    num_trainable_parameters = 0
    for p in model_video.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Video Params: {}".format(num_trainable_parameters))
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Fusion Params: {}".format(num_trainable_parameters))

    if ARGS.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=1e-4)
        optimizer_audio = optim.SGD(model_audio.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=1e-4)
        optimizer_video = optim.SGD(model_video.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=1e-4)
    elif ARGS.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)
        optimizer_audio = optim.Adam(model_audio.parameters(), lr=ARGS.lr)
        optimizer_video = optim.Adam(model_video.parameters(), lr=ARGS.lr)
    
    if ARGS.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)

    scaler = torch.cuda.amp.GradScaler()

    # resume model weight
    audio_ckpt = torch.load(ARGS.audio_path)
    model_audio.load_state_dict(audio_ckpt['net'])
    optimizer_audio.load_state_dict(audio_ckpt['optimizer'])

    video_ckpt = torch.load(ARGS.video_path)
    model_video.load_state_dict(video_ckpt['net'])
    optimizer_video.load_state_dict(video_ckpt['optimizer'])

    # load dataset
    train_dataset, val_dataset, test_dataset = load_dataset()
    train_loader, val_loader, test_loader = load_dataloader(train_dataset, val_dataset, test_dataset, BATCH_SIZE)
    test_idlist = test_dataset.get_idlist()
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader)*EPOCHS))


    BEST_VAL = 0
    best_model = model
    for epoch in range(EPOCHS):
        train_acc, train_loss, lr_rate = train(model, model_audio, model_video,
                                                    train_loader, optimizer, scheduler,
                                                    criterion, scaler, BATCH_SIZE)
        val_acc = validate(model, model_audio, model_video, val_loader, BATCH_SIZE)
        if BEST_VAL <= val_acc:
            save_checkpoint(val_acc, model, optimizer, epoch, logdir)
            best_model = model
            BEST_VAL = val_acc
            test(model, model_audio, model_video, test_loader, test_idlist, logdir, str(epoch).zfill(3))
    
    test(best_model, model_audio, model_video, test_loader, test_idlist, logdir, 'best')

