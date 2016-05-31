from skimage.util import pad
import time
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from model import compose_functions
from utils import dice
from tqdm import *
from batch import minibatch
from itertools import izip
from random import shuffle
import lasagne

data = np.load('train.npz', mmap_mode='r')

all_image_base_names = [ x for x in data.keys() if "mask" not in x ]

# Mix them up, TODO: give seed for reproducibility
shuffle(all_image_base_names)

patch_size = (416, 416)
num_epochs = 100
batch_size = 2

network, train_fn, val_fn = compose_functions(patch_size)

start_validation_index = int((len(all_image_base_names)/batch_size + len(all_image_base_names)%batch_size) * .9)
print("Start validation at: {}".format(start_validation_index))


# TODO: move all data modification to preprocessing
# TODO: minibatch
for epoch in range(0, num_epochs):
    dices = []
    index = 0 
    validation_errs = []
    train_errs = []
    for img_name_batch, i in izip(minibatch(all_image_base_names, batch_size=batch_size), tqdm(range(len(all_image_base_names)/batch_size + len(all_image_base_names)%batch_size ))):
        # Cool of the gpu a little
        if i % 500 == 0:
            time.sleep(60)
        image = np.array([data[x] for x in img_name_batch])
        image_mask = np.array([data["{}_mask".format(img_name)] for img_name in img_name_batch]) 

        if index < start_validation_index:
            err = train_fn(image, image_mask)
            train_errs.append(err)
        else:
            err, prediction = val_fn(image, image_mask)
            prediction[prediction > .5] = 1.
            prediction[prediction <= .5] = 0.
            #prediction = np.reshape(prediction, (img_name_batch.shape[0], patch_size[0], patch_size[1]))
            #image_mask = np.reshape(image_mask, (img_name_batch.shape[0], patch_size[0], patch_size[1]))
            dices.append(dice(prediction, image_mask))
            validation_errs.append(err)
        index += 1# img_name_batch.shape[0] 

    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #import pdb; pdb.set_trace() 

    print("Train ave loss: {}".format(np.average(train_errs)))
    print("Validation ave loss: {}".format(np.average(validation_errs)))
    print("Average dice: {}".format(np.average(dices)))
    # Cool off the gpu after an epoch
    time.sleep(200)
