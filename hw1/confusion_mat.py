#!/bin/python

import argparse
from operator import not_
import os
import pickle
import time

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize, StandardScaler

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Apply the MLP model to the testing videos;
# Output prediction class for each video

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_file")
  parser.add_argument("feat_dir")
  parser.add_argument("feat_dim", type=int)
  parser.add_argument("list_videos")
  parser.add_argument("output_file")
  parser.add_argument("--file_ext", default=".csv")

  return parser.parse_args()


if __name__ == '__main__':

  args = parse_args()

  # 1. load mlp model
  mlp = pickle.load(open(args.model_file, "rb"))

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category

  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.file_ext)
    # for videos with no audio, ignored in training
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))


  print("number of samples: %s" % len(feat_list))
  X = np.array(feat_list)
  y = np.array(label_list)

  print(f'X: {X.shape}')
  print(f'y: {y.shape}')

  # 3. Get predictions
  # (num_samples) with integer
  start = time.time()
  scaler = StandardScaler().fit(X)
  #X = scaler.transform(X)
  
  pred_classes = mlp.predict(X)
  end = time.time()
  print(f'Time elapsed for training: {end-start}')
  
  plot_confusion_matrix(mlp, X, y)  
  plt.savefig('soundnet.png')
