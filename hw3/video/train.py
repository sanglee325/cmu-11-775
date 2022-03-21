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
from models.mlp import MLPNetwork
from config import *


def train(model, train_loader, optimizer, scheduler, criterion, scaler, batch_size):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    model.train()

    num_correct = 0
    total_loss = 0
    total = 0

    for i, (x, y) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)
        
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

def validate(model, val_loader, batch_size):
    model.eval()

    num_correct = 0
    total = 0
    for i, (x, y) in enumerate(val_loader):

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            outputs = model(x)

        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        total += len(x)
        

    val_acc = 100 * num_correct / total
    print("Validation: {:.04f}%".format(100 * num_correct / total))

    return val_acc

def test(model, test_loader, test_idlist, logdir, name):
    model.eval()

    res = []
    total = 0
    for i, (x) in enumerate(test_loader):
        x = x.to(device)

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

    num_classes = 15
    input_size = 512
    
    model = MLPNetwork(input_size=input_size, num_classes=num_classes)

    model.to(device)

    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Params: {}".format(num_trainable_parameters))

    train_dataset, val_dataset, test_dataset = load_dataset()
    train_loader, val_loader, test_loader = load_dataloader(train_dataset, val_dataset, test_dataset, BATCH_SIZE)
    test_idlist = test_dataset.get_idlist()

    if ARGS.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=1e-4)
    elif ARGS.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader)*EPOCHS))
    
    if ARGS.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)

    scaler = torch.cuda.amp.GradScaler()

    BEST_VAL = 0
    best_model = model
    for epoch in range(EPOCHS):
        # You can add validation per-epoch here if you would like
        train_acc, train_loss, lr_rate = train(model, train_loader, optimizer, scheduler,
                                                     criterion, scaler, BATCH_SIZE)
        val_acc = validate(model, val_loader, BATCH_SIZE)
        if BEST_VAL <= val_acc:
            save_checkpoint(val_acc, model, optimizer, epoch, logdir)
            best_model = model
            BEST_VAL = val_acc
            test(model, test_loader, test_idlist, logdir, str(epoch).zfill(3))
    
    test(best_model, test_loader, test_idlist, logdir, 'best')

