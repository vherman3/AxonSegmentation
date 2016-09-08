import os
import pickle
import shutil
from scipy.misc import imread, imsave
from sklearn import preprocessing
from PIL import Image
import numpy as np


def extract_patch(img, mask, size):
    h, w = img.shape

    q_h, r_h = divmod(h, size)
    q_w, r_w = divmod(w, size)

    r2_h = size-r_h
    r2_w = size-r_w
    q2_h = q_h + 1
    q2_w = q_w + 1

    q3_h, r3_h = divmod(r2_h,q_h)
    q3_w, r3_w = divmod(r2_w,q_w)

    dataset = []
    pos = 0
    while pos+size<=h:
        pos2 = 0
        while pos2+size<=w:
            patch = img[pos:pos+size, pos2:pos2+size]
            patch_gt = mask[pos:pos+size, pos2:pos2+size]
            dataset.append([patch,patch_gt])
            pos2 = size + pos2 - q3_w
            if pos2 + size > w :
                pos2 = pos2 - r3_w

        pos = size + pos - q3_h
        if pos + size > h:
            pos = pos - r3_h
    return dataset


def build_data(path_data, trainRatio = 0.80, folder_number=1):

    i = 0
    for root in os.listdir(path_data)[1:]:
        subpath_data = os.path.join(path_data, root)
        for data in os.listdir(subpath_data):
            if 'image' in data:
                img = imread(os.path.join(subpath_data, data), flatten=False, mode='L')
            elif 'mask' in data:
                mask = preprocessing.binarize(imread(os.path.join(subpath_data, data), flatten=False, mode='L'), threshold=125)
        if i ==0:
            patches = extract_patch(img, mask, 256)
            print len(patches)
        else:
            patches += extract_patch(img, mask, 256)
            print len(patches)
        i+=1

    testRatio = 1-trainRatio
    size_test = int(testRatio*len(patches))

    patches_train = patches[:-size_test]
    patches_test = patches[-size_test:]

    folder = 'dataset'
    if not os.path.exists(folder):
        os.makedirs(folder)

    folder_train = folder+''+'/Train'+str(folder_number)

    if os.path.exists(folder_train):
        shutil.rmtree(folder_train)
    if not os.path.exists(folder_train):
        os.makedirs(folder_train)

    folder_test = folder+''+'/Test'+str(folder_number)

    if os.path.exists(folder_test):
        shutil.rmtree(folder_test)
    if not os.path.exists(folder_test):
        os.makedirs(folder_test)


    j = 0
    for patch in patches_train:
        imsave(folder_train+'/image_%s.jpeg'%j, patch[0],'jpeg')
        imsave(folder_train+'/classes_%s.jpeg'%j, patch[1].astype(int),'jpeg')
        j+=1

    k=0
    for patch in patches_test:
        imsave(folder_test+'/image_%s.jpeg'%k, patch[0],'jpeg')
        imsave(folder_test+'/classes_%s.jpeg'%k, patch[1].astype(int),'jpeg')
        k+=1

build_data('/Users/viherm/Desktop/CARS', trainRatio = 0.80, folder_number = 1)

