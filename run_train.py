from skimage.util import pad
import time
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from model import compose_functions
from utils import dice
from tqdm import *
from batch import minibatch
from itertools import izip


data = np.load('train.npz', mmap_mode='r')

all_image_base_names = [ x for x in data.keys() if "mask" not in x ]

patch_size = (33, 33)
num_epochs = 1
batch_size = 4000

network, train_fn, val_fn = compose_functions(patch_size)

index = 0
start_validation_index = int(len(all_image_base_names) * .8)

train_err = 0.
validation_err = 0.

for img_name, i in izip(iter(all_image_base_names), tqdm(range(len(all_image_base_names)))):
    image, image_mask = data[img_name], data["{}_mask".format(img_name)] 

    # trim to square
    difference = (image.shape[1] - image.shape[0])/2
    image = image[:, difference:image.shape[1]-difference]
    image_mask = image_mask[:, difference:image_mask.shape[1]-difference]

    small_padded_img = pad(image, patch_size[0]/2, mode='wrap')
    patches = extract_patches_2d(image, patch_size)
    mask_vals = []
    i = 0
    for patch in patches:
        mask_vals.append(image_mask[i/image_mask.shape[0], i%image_mask.shape[1]])
        i += 1
    mask_val_array = np.array(mask_vals)
    patches = np.reshape(patches, (patches.shape[0], 1, patches.shape[1], patches.shape[2]))

    for patch_batch, mask_batch, i in izip(minibatch(patches, batch_size), minibatch(mask_val_array, batch_size), tqdm(range(patches.shape[0]/batch_size))):
        if index < start_validation_index:
            #start_time = time.time()
            train_err += train_fn(patch_batch, mask_batch)
            #end_time = time.time()
            #print '%s function took %0.3f ms' % ("train", (end_time-start_time)*1000.0)
        else:
            err, prediction = val_fn(patches, mask_vals)
            dices.append(dice(prediction, mask_vals)) 
            validation_err += err

print("Train error: {}".format(train_err))
print("Validation error: {}".format(validation_err))
print("Average dice: {}".format(sum(dices)/len(dices)))