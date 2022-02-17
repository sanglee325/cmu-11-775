#!/bin/python

import argparse
import os

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help='Path to the txt file with list of file names and labels')
    parser.add_argument("--pred_file", type=str,
                        help='Path to the file where the seleted MFCC samples will be stored')


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    y = pd.read_csv(args.input_file)
    pred = pd.read_csv(args.pred_file)
    
    correct = (y['Category'] == pred['Category'])
    
    acc = sum(correct) / len(y)
    print(acc)