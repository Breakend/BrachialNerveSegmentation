#!/usr/bin/python3
from progressbar import *
import numpy as np
import matplotlib.pyplot as plt  # Just so we can visually confirm we have the same images
import argparse
import os

# CLI args
parser = argparse.ArgumentParser(description='Preprocess the image data into numpy format')
parser.add_argument('images_directory_path')
parser.add_argument('output_file_name')
args = parser.parse_args()

# Inits
data_dict = {}


print("Loading data into memory...")

for dirname, dirnames, filenames in os.walk(args.images_directory_path):
    pbar = ProgressBar(maxval=len(filenames))
    i = 0
    pbar.start()
    # print path to all filenames.
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        img = plt.imread(file_path)
        # strip out tiff ending
        filename = filename[:-4]
        data_dict[filename] = img
        pbar.update(i)
        i += 1
pbar.finish()

print("Saving data to compressed npz file with name {}... This might take a while...".format(args.output_file_name))
np.savez_compressed(args.output_file_name, **data_dict)

del data_dict

