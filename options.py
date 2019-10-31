import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Environmental Sound Classification Using CNN")

    # Data
    parser.add_argument('--baseline', action='store_true', help='Using the baseline SVM approach')

    opts = parser.parse_args(args)
    
    return opts
