import numpy as np
from model import compose_prediction_functions
from RunLengthEncoder import encode_run_length
import argparse
from itertools import izip
from scipy.spatial import ConvexHull
from tqdm import *
import lasagne
import csv
import os
import cv2

patch_size = (416,416)

network, val_fn = compose_prediction_functions(patch_size)

with np.load('model.npz') as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

data = np.load('test.npz', mmap_mode='r')
all_image_base_names = [ x for x in data.keys() if "mask" not in x ]

index = 0
start_validation_index = int(len(all_image_base_names) * .9)
print("Start validation at: {}".format(start_validation_index))

predictions = {}
# TODO: move all data modification to preprocessing
# TODO: minibatch
for img_name, i in izip(iter(all_image_base_names), tqdm(range(len(all_image_base_names)))):
    image = data[img_name]
    image = np.reshape(image, (1,1, image.shape[0], image.shape[1]))

    prediction = val_fn(image)
    prediction = np.reshape(prediction, patch_size)
    prediction[prediction < .8] = 0.
    prediction[prediction >= .8] = 1.0
    kernel = np.ones((5,5), np.uint8)
    opened = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
    opened = cv2.erode(opened,kernel,iterations = 3)
    opened = cv2.dilate(opened,kernel,iterations = 2)
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

    predictions[img_name] = opened
    
# write to submission file
print('Writing submission to file...')
fi = csv.reader(open('sample_submission.csv'))
f = open('submission.csv', 'w')
fo = csv.writer(f, lineterminator='\n')
fo.writerow(fi.next())
for line in fi:
    idx = line[0]
    out = [idx]
    out.append(encode_run_length(predictions[idx]))
    fo.writerow(out)

f.close()

