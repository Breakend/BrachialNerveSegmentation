#!/usr/bin/python3
from progressbar import *
import numpy as np
import matplotlib.pyplot as plt  # Just so we can visually confirm we have the same images
import argparse
from scipy import ndimage
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

def rotation_augmentation(X, Y, angle_range):
    X_rot = np.copy(X)
    Y_rot = np.copy(Y)
    angle = np.random.randint(-angle_range, angle_range)
    for j in range(X.shape[0]):
        X_rot[j] = ndimage.rotate(X[j], angle, reshape=False, order=1)
        Y_rot[j] = ndimage.rotate(Y[j], angle, reshape=False, order=1)
    return X_rot, Y_rot

def shift_augmentation(X, Y, h_range, w_range):
    X_shift = np.copy(X)
    Y_shift = np.copy(Y)
    size = X.shape[1:]
    h_random = np.random.rand() * h_range * 2. - h_range
    w_random = np.random.rand() * w_range * 2. - w_range
    h_shift = int(h_random * size[0])
    w_shift = int(w_random * size[1])
    for j in range(X.shape[0]):
        X_shift[j] = ndimage.shift(X[j], (h_shift, w_shift), order=0)
        Y_shift[j] = ndimage.shift(Y[j], (h_shift, w_shift), order=0)
    return X_shift, Y_shift

def trim_n_stuff(image):
    # trim to square
    difference = (image.shape[1] - image.shape[0])/2
    image = np.true_divide(image[:, difference:image.shape[1]-difference], np.float32(255.))

    # trim the difference
    x_diff = (image.shape[0] - patch_size[0])/2
    y_diff = (image.shape[1] - patch_size[1])/2
    image = image[x_diff:image.shape[0]-x_diff, y_diff:image.shape[1]-y_diff]
    image = np.reshape(image, (1, image.shape[0], image.shape[1]))
    return image

for dirname, dirnames, filenames in os.walk(args.images_directory_path):
    pbar = ProgressBar(maxval=len(filenames))
    i = 0
    pbar.start()
    # print path to all filenames.
    for filename in filenames:
        import pdb; pdb.set_trace()
        file_path = os.path.join(dirname, filename)
        untrimmed = image = plt.imread(file_path)
        untrimmed = np.reshape(untrimmed, (1, untrimmed.shape[0], untrimmed.shape[1]))

        image = trim_n_stuff(image)

        if "mask" not in filename:
            image_mask_path = os.path.join(dirname, filename[:-4] + "_mask.tif")
            image_mask = plt.imread(image_mask_path)
            image_mask = np.reshape(image_mask, (1, image_mask.shape[0], image_mask.shape[1]))
            image_rotated, image_mask_rotated = rotation_augmentation(untrimmed, image_mask, 15)
            image_shifted, image_mask_shifted = shift_augmentation(untrimmed, image_mask, .05, .05) 
            data_dict[filename[:-4]+"_rotated"] = trim_n_stuff(np.reshape(image_rotated, (image_rotated.shape[1], image_rotated.shape[2])))
            data_dict[filename[:-4]+"_rotated_mask"] = trim_n_stuff(np.reshape(image_mask_rotated, (image_mask_rotated.shape[1], image_mask_rotated.shape[2])))
            data_dict[filename[:-4]+"_shifted"] = trim_n_stuff(np.reshape(image_shifted, (image_rotated.shape[1], image_rotated.shape[2])))
            data_dict[filename[:-4]+"_shifted_mask"] = trim_n_stuff(np.reshape(image_mask_shifted, (image_rotated.shape[1], image_rotated.shape[2])))

        # strip out tiff ending
        filename = filename[:-4]
        data_dict[filename] = image
        pbar.update(i)
        i += 1
pbar.finish()

print("Saving data to compressed npz file with name {}... This might take a while...".format(args.output_file_name))
np.savez_compressed(args.output_file_name, **data_dict)

del data_dict

