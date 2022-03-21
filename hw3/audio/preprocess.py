#!/bin/python
import argparse
import os

import librosa
from tqdm import tqdm

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str,
                        help='Path to the .wav file')

    parser.add_argument('-o', '--output_dir', type=str,
                        help='Path to the file where the mfcc will be stored')

    return parser.parse_args()

def get_mfcc(file_list, input_dir, output_dir):
    for wav_file in tqdm(file_list):
        file_path = os.path.join(input_dir, wav_file)
        signal, sr = librosa.load(file_path, sr=16000)
        
        if len(signal) < 10*sr:
            zpad = 10*sr - len(signal)
            signal = np.pad(signal, (0,zpad), 'constant', constant_values=0)
        elif len(signal) > 10*sr:
            signal = signal[:10*sr]

        MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
 

        filename = wav_file.split('.')
        
        mfcc_csv = filename[0] + ".mfcc.csv"
        output_path = os.path.join(output_dir, mfcc_csv)
        np.savetxt(output_path, MFCCs, delimiter=';')

    return MFCCs
    
if __name__ == '__main__':
    args = parse_args()

    file_list = os.listdir(args.input_dir)
    MFCCs = get_mfcc(file_list, args.input_dir, args.output_dir)
    

