import numpy as np
from RunLengthEncoder import encode_run_length
import argparse
from itertools import izip
import matplotlib.pyplot as plt  # Just so we can visually confirm we have the same images
import cv2
import csv
import os
import random

patch_size = (416,416)

# CLI args
parser = argparse.ArgumentParser(description='Preprocess the image data into numpy format')
parser.add_argument('images_directory_path')
args = parser.parse_args()

for dirname, dirnames, filenames in os.walk(args.images_directory_path):
    for filename in filenames:
        if "mask" in filename:
            file_path = os.path.join(dirname, filename)
            ground_truth = plt.imread(file_path)
            fig = plt.figure(1)
            fig.canvas.set_window_title('truth')
            plt.imshow(ground_truth, interpolation='nearest')
            plt.show()
