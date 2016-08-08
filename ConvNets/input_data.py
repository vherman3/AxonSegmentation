from scipy.misc import imread
from sklearn import preprocessing
import numpy as np
import random

import matplotlib.pyplot as plt

class input_data:
    def __init__(self, type = 'train'):
        if type == 'train' :
            self.path = 'dataset/Train/'
            self.set_size = 34414
        if type == 'test':
            self.path = 'dataset/Test/'
            self.set_size = 17209
        self.size_image = 256
        self.batch_start = 0
        self.n_labels = 2

    def extract_batch(self, start, size):
        batch_x = []
        for i in range(start, start + size):
            image = imread(self.path + 'image_%s.jpeg'%i, flatten=False, mode='L')
            category = preprocessing.binarize(imread(self.path + 'classes_%s.jpeg'%i, flatten=False, mode='L'), threshold = 125)
            batch_x.append(image)
            if i == start:
                batch_y = category.reshape(-1,1)
            else:
                batch_y = np.concatenate((batch_y, category.reshape(-1, 1)), axis=0)
        batch_y = np.concatenate((np.invert(batch_y)/255, batch_y), axis = 1)

        return [np.asarray(batch_x), batch_y]

    def next_batch(self, batch_size, rnd = False):
        batch_start = self.batch_start
        self.batch_start+=batch_size
        if random :
            self.batch_start = random.randint(0,self.set_size - batch_size)
        return self.extract_batch(batch_start, batch_size)

    def read_batch(self, batch_y, size_batch):
        images = batch_y.reshape(size_batch, self.size_image, self.size_image, self.n_labels)
        return images

#
#
# data_train = input_data('train')
# batch_x, batch_y = data_train.next_batch(20)
# mask = data_train.read_batch(batch_y,20)
#
# print batch_x.shape
#
# im = batch_x[0,:,:]
# mask = mask[0, :, :,1]
#
# plt.figure()
# plt.imshow(im, cmap=plt.get_cmap('gray'))
# plt.hold(True)
# plt.imshow(mask, alpha=0.7)
# plt.show()
#
# print mask
#






