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
patch_size = (416,416)

print("Loading data into memory...")

for dirname, dirnames, filenames in os.walk(args.images_directory_path):
    pbar = ProgressBar(maxval=len(filenames))
    i = 0
    pbar.start()
    # print path to all filenames.
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        image = plt.imread(file_path)
        # trim to square
        difference = (image.shape[1] - image.shape[0])/2
        image = np.true_divide(image[:, difference:image.shape[1]-difference], np.float32(255.))

        # trim the difference
        x_diff = (image.shape[0] - patch_size[0])/2
        y_diff = (image.shape[1] - patch_size[1])/2
        image = image[x_diff:image.shape[0]-x_diff, y_diff:image.shape[1]-y_diff]

        # strip out tiff ending
        filename = filename[:-4]
        data_dict[filename] = image
        pbar.update(i)
        i += 1
pbar.finish()

print("Saving data to compressed npz file with name {}... This might take a while...".format(args.output_file_name))
np.savez_compressed(args.output_file_name, **data_dict)

del data_dict

