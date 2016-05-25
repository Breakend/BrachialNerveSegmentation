import numpy as np
from model import compose_prediction_functions
from RunLengthEncoder import encode_run_length
import argparse
from itertools import izip
from tqdm import *
import lasagne
import csv
import os

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
    prediction[prediction < .97] = 0.
    prediction[prediction >= .97] = 1.0
    predictions[img_name] = prediction
    
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

