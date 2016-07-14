import numpy as np
import pickle
from skimage import exposure
from sklearn.feature_extraction.image import extract_patches_2d
import os.path
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.transform import rescale
import random
from sklearn import preprocessing
from scipy.stats import bernoulli
import math
import numpy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from rotation_crop import crop_around_center,rotate_image, largest_rotated_rect
import gzip
from random import shuffle
import scipy


folder = os.path.abspath(os.path.join(os.pardir))

data = pickle.load(open(folder +"/data/groundTruth.pkl", "rb"))

img = data['image']
img = exposure.equalize_hist(img)
mask = data['mask']

h, w = img.shape

#######################################################################################################################
#                                                     Some augmentation strategies                                    #
#######################################################################################################################


def extract_patch(img,mask,size):
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

            if pos2 + size+10 <=w:
                patch = img[pos:pos+size, pos2+10:pos2+size+10]
                patch_gt = mask[pos:pos+size, pos2+10:pos2+size+10]
                dataset.append([patch,patch_gt])

            if pos2 - 10>=0:
                patch = img[pos:pos+size, pos2-10:pos2+size-10]
                patch_gt = mask[pos:pos+size, pos2-10:pos2+size-10]
                dataset.append([patch,patch_gt])

            if pos + size+10 <=h:
                patch = img[pos+10:pos+size+10, pos2:pos2+size]
                patch_gt = mask[pos+10:pos+size+10, pos2:pos2+size]
                dataset.append([patch,patch_gt])

            if pos - 10 >= 0:
                patch = img[pos-10:pos+size-10, pos2:pos2+size]
                patch_gt = mask[pos-10:pos+size-10, pos2:pos2+size]
                dataset.append([patch,patch_gt])


            pos2 = size + pos2 - q3_w
            if pos2 + size > w :
                pos2 = pos2 - r3_w

        pos = size + pos - q3_h
        if pos + size > h:
            pos = pos - r3_h
    return dataset


def flipped_lr(dataset):
    flipped = []
    for i in range(len(dataset)):
        s = np.random.binomial(1, 0.7, 1)
        if s == 1 :
            image = dataset[i][0]
            gt = dataset[i][1]
            flipped.append([np.fliplr(image), np.fliplr(gt)])
    return flipped


def flipped_ud(dataset):
    flipped = []
    for i in range(len(dataset)):
        s = np.random.binomial(1, 0.7, 1)
        if s == 1 :
            image = dataset[i][0]
            gt = dataset[i][1]
            flipped.append([np.flipud(image), np.flipud(gt)])
    return flipped


def resc(img, mask, size):
    s = np.random.uniform(0.5, 2.0, 30)
    data_rescale=[]
    for scale in s:
        image_rescale = rescale(img, scale)
        mask_rescale = rescale(mask, scale)
        mask_rescale = preprocessing.binarize(np.array(mask_rescale), threshold=0.001)

        patches = extract_patch(image_rescale, mask_rescale, size)
        data_rescale+= random.sample(patches, 3)

    return data_rescale


def elastic_transform(image, gt, alpha, sigma, random_state=None):

    if random_state is None:
        random_state = numpy.random.RandomState(None)

    shape = image.shape

    d = 4
    sub_shape = (shape[0]/d, shape[0]/d)

    deformations_x = random_state.rand(*sub_shape) * 2 - 1
    deformations_y = random_state.rand(*sub_shape) * 2 - 1

    deformations_x = np.repeat(np.repeat(deformations_x, d, axis=1), d, axis = 0)
    deformations_y = np.repeat(np.repeat(deformations_y, d, axis=1), d, axis = 0)

    dx = gaussian_filter(deformations_x, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(deformations_y, sigma, mode="constant", cval=0) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))

    elastic_image = map_coordinates(image, indices, order=1).reshape(shape)
    elastic_gt = map_coordinates(gt, indices, order=1).reshape(shape)
    elastic_gt = preprocessing.binarize(np.array(elastic_gt), threshold=0.5)



    #plt.figure()
    #plt.quiver(x, y, dx, dy)
    #plt.show()

    return [elastic_image, elastic_gt]


def elastic(dataset):
    elas = []
    for i in range(len(dataset)):
        for j in range(5):
            elas.append(elastic_transform(dataset[i][0], dataset[i][1], alpha = 7, sigma = 4))
            elas.append(elastic_transform(dataset[i][0], dataset[i][1], alpha = 10, sigma = 4))
        print 'iteration : %s/%s'%(i,len(dataset))
    return elas


def random_rotation(img, mask,size):
    rotations = []
    for angle in np.random.uniform(5, 89, 30):

        image_rotated = rotate_image(img, angle)
        gt_rotated = rotate_image(mask, angle)

        rect = largest_rotated_rect(w, h, math.radians(angle))

        image_rotated_cropped = crop_around_center(image_rotated, *rect)
        gt_rotated_cropped = crop_around_center(gt_rotated, *rect)

        patches = extract_patch(image_rotated_cropped, gt_rotated_cropped, size)
        rotations+=patches
    return rotations

#######################################################################################################################
#                                                     Augmentation and visualization                                  #
#######################################################################################################################

dataset = extract_patch(img,mask,256)
dataset += resc(img, mask, 256)
print 'Size after rescaling ', len(dataset)
dataset += random_rotation(img, mask, 256)
print ' after rotations ', len(dataset)
dataset += flipped_lr(dataset)
dataset += flipped_ud(dataset)
print ' after flipped ', len(dataset)
dataset += elastic(dataset)
print ' after elastric deformations ', len(dataset)

shuffled = random.sample(dataset, len(dataset))

data_train = shuffled[:2*(len(dataset)//3)]
data_test = shuffled[2*(len(dataset)//3):]


# for i in range(len(data_train)):
#     plt.figure(i)
#     plt.imshow(data_train[i][0], cmap=plt.get_cmap('gray'))
#     plt.hold(True)
#     plt.imshow(data_train[i][1], alpha=0.7)
#     plt.show()
#
#     print np.unique(data_train[i][1])


folder = 'dataset'
if not os.path.exists(folder):
    os.makedirs(folder)

folder_train = folder+''+'/Train'
if not os.path.exists(folder_train):
    os.makedirs(folder_train)

i=0
for image in data_train:
    scipy.misc.imsave(folder_train+'/image_%s.jpeg'%i, image[0],'jpeg')
    scipy.misc.imsave(folder_train+'/classes_%s.jpeg'%i, image[1].astype(int),'jpeg')
    i+=1

folder_test = folder+''+'/Test'
if not os.path.exists(folder_test):
    os.makedirs(folder_test)

i=0
for image in data_test:
    scipy.misc.imsave(folder_test+'/image_%s.jpeg'%i, image[0],'jpeg')
    scipy.misc.imsave(folder_test+'/classes_%s.jpeg'%i, image[1].astype(int),'jpeg')
    i+=1



