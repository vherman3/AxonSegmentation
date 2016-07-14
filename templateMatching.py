import pickle
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import scipy
import numpy as np
import astropy
from skimage import exposure
from astropy.convolution import TrapezoidDisk2DKernel, Ring2DKernel
from astropy.convolution import convolve, convolve_fft
from skimage.feature import match_template


data = pickle.load(open("data/groundTruth.pkl", "rb"))
img = data['image']
img = exposure.equalize_hist(img)


def template_matching(image,kernel_radius, gRatio):

    r = gRatio*kernel_radius
    y,x = np.ogrid[-kernel_radius: kernel_radius+1, -kernel_radius: kernel_radius+1]
    D = x**2+y**2 <= kernel_radius**2
    d = x**2+y**2 <= r**2
    template = (D-d)*255

    result = match_template(image, template)

    return [template, result]

radius = [10, 20, 30, 40, 100]

i = 1
for r in radius :

    template, result = template_matching(img, r, 0.7)
    plt.figure(i+1)
    plt.imshow(template, interpolation='none', origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.colorbar()
    plt.title('diameter : %s'%r)

    plt.figure(i)
    plt.imshow(result)
    plt.title('diameter : %s'%r)
    i+=1
    plt.show()
