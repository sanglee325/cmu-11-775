#!/bin/python

import argparse
import os

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        help='Path to the txt file with list of file names and labels')

    parser.add_argument("--output_path", type=str,
                        help='Path to the file where the seleted MFCC samples will be stored')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    

