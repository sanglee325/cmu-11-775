#!/bin/python
# Randomly select MFCC frames

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
                        help='Path to the file where the seleted MFCC samples will be stored')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input_file)
    
    eval_df = df.sample(frac=args.ratio, random_state=1000)
    train_df = df.drop(eval_df.index)

    eval_df = eval_df.sort_index()
    eval_df = eval_df.reset_index()
    eval_df.to_csv(os.path.join(args.output_path, 'eval_r01.csv'),index=False)
    train_df.to_csv(os.path.join(args.output_path, 'train_r01.csv'),index=False)
