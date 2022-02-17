#!/bin/python
# Create train and validation dataset

import argparse
import os

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help='Path to the txt file with list of file names and labels')

    parser.add_argument("--ratio", type=float, default=0.1,
                        help='Proportion of data that will be set as validation')

    parser.add_argument("--output_path", type=str,
                        help='Path to the file where created train/validation list will be stored')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. load input train_val.csv
    df = pd.read_csv(args.input_file)
    
    # 2. sample train and validation data based on given ratio
    val_df = df.sample(frac=args.ratio, random_state=1000)
    train_df = df.drop(val_df.index)

    val_df = val_df.sort_index()
    
    # 3. save files
    val_df.to_csv(os.path.join(args.output_path, 'validation.csv'),index=False)
    train_df.to_csv(os.path.join(args.output_path, 'train.csv'),index=False)
