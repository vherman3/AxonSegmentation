from skimage import exposure
from skimage.transform import rescale
import numpy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from rotation_crop import rotate_image
from scipy.misc import imread
from sklearn import preprocessing
import numpy as np
import random
import os
import matplotlib.pyplot as plt


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
            pos2 = size + pos2 - q3_w
            if pos2 + size > w :
                pos2 = pos2 - r3_w

        pos = size + pos - q3_h
        if pos + size > h:
            pos = pos - r3_h
    return dataset


def flipped(patch):
    s = np.random.binomial(1, 0.5, 1)
    image = patch[0]
    gt = patch[1]
    if s == 1 :
        image, gt = [np.fliplr(image), np.fliplr(gt)]
    s = np.random.binomial(1, 0.5, 1)
    if s == 1:
        image, gt=[np.flipud(image), np.flipud(gt)]
    return [image,gt]


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

    return [elastic_image, elastic_gt]


def elastic(patch):
    alpha = random.choice([1,2,3,4,5,6,7,8,9])
    elas = elastic_transform(patch[0],patch[1], alpha = alpha, sigma = 4)
    return elas


def shifting(patch, size_shift):

    img = np.pad(patch[0],size_shift,mode = "reflect")
    mask = np.pad(patch[1],size_shift,mode = "reflect")
    begin_h = np.random.randint(2*size_shift-1)
    begin_w = np.random.randint(2*size_shift-1)
    shifted_image = img[begin_h:,begin_w:]
    shifted_mask = mask[begin_h:,begin_w:]

    return [shifted_image,shifted_mask]



def resc(patch):
    s = random.choice([0.5, 0.75, 1.0, 1.5, 2.0])
    data_rescale=[]
    for scale in s:

        image_rescale = rescale(patch[0], scale)
        mask_rescale = rescale(patch[1], scale)
        s_r = mask_rescale.shape[0]
        q_h, r_h = divmod(256-s_r,2)

        if q_h > 0 :
            image_rescale = np.pad(image_rescale,(q_h, q_h+r_h), mode = "reflect")
            mask_rescale = np.pad(mask_rescale,(q_h, q_h+r_h), mode = "reflect")
        else :
            patches = extract_patch(image_rescale,mask_rescale, 256)
            i = np.random.randint(len(patches), size=1)
            image_rescale,mask_rescale = patches[i]

        mask_rescale = preprocessing.binarize(np.array(mask_rescale), threshold=0.001)
        data_rescale = [image_rescale, mask_rescale]

    return data_rescale


def random_rotation(img, mask, size):

    img = np.pad(img,190,mode = "reflect")
    mask = np.pad(mask,190,mode = "reflect")
    angle = np.random.uniform(5, 89, 1)

    image_rotated = rotate_image(img, angle)
    s_p = image_rotated.shape[0]
    center = int(float(s_p)/2)
    gt_rotated = rotate_image(mask, angle)

    image_rotated_cropped = image_rotated[center-128:center+128, center-128:center+128]
    gt_rotated_cropped = gt_rotated[center-128:center+128, center-128:center+128]

    return [image_rotated_cropped, gt_rotated_cropped]


def augmentation(patch,size = 256):
    patch = shifting(patch,10)
    patch = random_rotation(patch[0], patch[1], size = 256)
    patch = elastic(patch)
    patch = flipped(patch)
    return patch


#######################################################################################################################
#                                                     Some augmentation strategies                                    #
#######################################################################################################################

class input_data:
    def __init__(self, trainingset_path, type = 'train'):
        if type == 'train' :
            self.path = trainingset_path+'/Train/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])
        if type == 'test':
            self.path = trainingset_path+'/Test/'
            self.set_size = len([f for f in os.listdir(self.path) if ('image' in f)])

        self.size_image = 256
        self.n_labels = 2
        self.batch_start = 0

    def set_batch_start(self, start = 0):
        self.batch_start = 0

    def next_batch(self, batch_size, rnd = False, augmented_data = True):
        batch_x = []
        for i in range(batch_size) :
            if rnd :
                indice = random.choice(range(self.set_size))
            else :
                indice = self.batch_start
                self.batch_start += 1
                if self.batch_start >= self.set_size:
                    self.batch_start= 0

            image = imread(self.path + 'image_%s.jpeg' % indice, flatten=False, mode='L')
            image = exposure.equalize_hist(image)
            image = (image - np.mean(image))/np.std(image)

            category = preprocessing.binarize(imread(self.path + 'classes_%s.jpeg' % indice, flatten=False, mode='L'), threshold=125)

            if augmented_data :
                [image, category] = augmentation([image, category])

            # plt.figure(1)
            # plt.imshow(image, cmap=plt.get_cmap('gray'))
            # plt.hold(True)
            # plt.imshow(category, alpha=0.7)
            # plt.show()

            batch_x.append(image)
            if i == 0:
                batch_y = category.reshape(-1,1)
            else:
                batch_y = np.concatenate((batch_y, category.reshape(-1, 1)), axis=0)
        batch_y = np.concatenate((np.invert(batch_y)/255, batch_y), axis = 1)

        return [np.asarray(batch_x), batch_y]

    def read_batch(self, batch_y, size_batch):
        images = batch_y.reshape(size_batch, self.size_image, self.size_image, self.n_labels)
        return images


# data_train = input_data(dataset_number=1,type = 'train')
# for i in range(1,100):
#     data_train.next_batch(batch_size=1, rnd = False, augmented_data=True)

