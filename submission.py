from __future__ import print_function

import numpy as np
import cv2
from data import image_cols, image_rows
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt

def prep(img, should_hull=True):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows))
    kernel = np.ones((5,5), np.uint8)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
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

    if should_hull:
        return opened
    else:
        return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    from data import load_test_data
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load('imgs_mask_test.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    submission()
