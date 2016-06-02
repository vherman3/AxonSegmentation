import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage import transform as tf
import math

data = pickle.load(open("data/data2.pkl", "rb"))
img = data['image']
mask = data['mask']

img = img[:800, :800]
h,w = img.shape
#######################################################################################################################
#                                                      Rotations                                                      #
#######################################################################################################################


for i in range(1, 19):

    transf = tf.rotate(img, angle=i*10, resize=False)
    plt.figure(i)
    plt.imshow(transf, cmap=plt.get_cmap('gray'))

    transf = tf.rotate(img, angle=i*10, resize=False, center = (float(h)/2+50, float(w)/2))

    plt.figure(i*2)
    plt.imshow(transf, cmap=plt.get_cmap('gray'))


plt.show()


