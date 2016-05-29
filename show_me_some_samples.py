import numpy as np
from model import compose_prediction_functions
from RunLengthEncoder import encode_run_length
import argparse
from itertools import izip
from tqdm import *
import lasagne
import cv2
import csv
from scipy.spatial import ConvexHull
import os
import random

patch_size = (416,416)

network, val_fn = compose_prediction_functions(patch_size)

with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

data = np.load('train.npz', mmap_mode='r')
all_image_base_names = [ x for x in data.keys() if "mask" not in x ]

index = 0

predictions = {}
# TODO: move all data modification to preprocessing
# TODO: minibatch
img_name = random.choice(all_image_base_names)
image = data[img_name]
ground_truth = data[img_name + "_mask"]
image = np.reshape(image, (1, 1, patch_size[0], patch_size[1]))

prediction = val_fn(image)
prediction = np.reshape(prediction[0], patch_size)
prediction[prediction > .5] = 1.
prediction[prediction <= .5] = 0.

kernel = np.ones((5,5), np.uint8)
opened = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
opened = cv2.erode(opened,kernel,iterations = 1)
opened = cv2.dilate(opened,kernel,iterations = 1)
convex_hull = None
if len(np.transpose(np.nonzero(opened))) > 0:
    convex_hull = ConvexHull(np.transpose(np.nonzero(opened)))
    if convex_hull.area > 150:
        hull = np.squeeze(cv2.convexHull(np.transpose(np.nonzero(opened))))
        hull = np.fliplr(hull)
        mask = np.zeros_like(opened)
        cv2.drawContours(mask, [hull], 0, 1, thickness=-1)
        opened = mask
    else:
        opened = np.zeros_like(opened)

if convex_hull:
    print("Simplices: {}".format(convex_hull.simplices.shape))
    print("Area: {}".format(convex_hull.area))

opened *= 255
prediction *= 255

from matplotlib import pyplot as plt
fig = plt.figure(0)
fig.canvas.set_window_title('prediction')
plt.imshow(prediction, interpolation='nearest')
fig = plt.figure(1)
fig.canvas.set_window_title('truth')
plt.imshow(np.reshape(ground_truth, patch_size), interpolation='nearest')
fig = plt.figure(2)
fig.canvas.set_window_title('opened')
plt.imshow(opened, interpolation='nearest')
plt.show()
