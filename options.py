import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Environmental Sound Classification Using CNN")

    # Data
    parser.add_argument('--sr', type=int, default=16000, help='Sampling rate for audio samples')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', default="./data/")

    # Model
    parser.add_argument('--baseline', action='store_true', help='Using the baseline SVM approach')

    opts = parser.parse_args(args)

    return opts
