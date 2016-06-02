import numpy as np
from scipy import ndimage as ndi
from skimage.filters.rank import entropy, gradient
from skimage.filters import sobel, canny
from skimage.morphology import disk
from sklearn.feature_extraction.image import extract_patches_2d
import pickle
from skimage import exposure
import matplotlib.pyplot as plt
from scipy.signal import decimate


def filter_bank(img, coeff_resolution):
    """
    Calculates the responses of an image to M filters.
    Returns 2-d array of the vectorial responses
    """

    h, w = img.shape

    im = np.reshape(img, (h*w, 1))

    e1 = np.reshape(entropy(img, disk(coeff_resolution*5)), (h*w, 1))
    e2 = np.reshape(entropy(img, disk(coeff_resolution*8)), (h*w, 1))
    e3 = np.reshape(entropy(img, disk(coeff_resolution*10)), (h*w, 1))

    g1 = np.reshape(gradient(img, disk(1)), (h*w, 1))
    g2 = np.reshape(gradient(img, disk(coeff_resolution*3)), (h*w, 1))
    g3 = np.reshape(gradient(img, disk(coeff_resolution*5)), (h*w, 1))

    m1 = np.reshape(ndi.maximum_filter(256-img, size=coeff_resolution*2, mode='constant'), (h*w, 1))
    m2 = np.reshape(ndi.maximum_filter(256-img, size=coeff_resolution*4, mode='constant'), (h*w, 1))
    m3 = np.reshape(ndi.maximum_filter(256-img, size=coeff_resolution*10, mode='constant'), (h*w, 1))

    #c = np.reshape(canny(img), (h*w, 1))
    s = np.reshape(sobel(img), (h*w, 1))

    return np.column_stack((im, e1, e2, e3, g1, g2, g3, m1, m2, m3, s))


def features(img, size_patch, coeff_resolution=1.0):
    """
    For each pixel of an image, takes filter response into a centered patch.
    Returns for each pixel a vector of shape (1,n_filters*size_patch)
    """

    h, w = img.shape
    resp = filter_bank(img, coeff_resolution)
    nb_filters = resp.shape[1]

    for f in range(nb_filters):
        im_resp = np.reshape(resp[:, f], (h, w))
        patches = extract_patches_2d(np.lib.pad(im_resp, (1, 1), 'edge'), (size_patch, size_patch))
        if f == 0:
            patches_raw = patches.reshape(h*w, -1)
        else:
            patches_raw = np.concatenate((patches_raw, patches.reshape(h*w, -1)), 1)

    return patches_raw



#######################################################################################################################
#                                                      Disply features                                                #
#######################################################################################################################

#
# data = pickle.load(open("data/data2.pkl", "rb"))
# img = data['image']
# img_reduced = decimate(img, 2,axis = 1)
# img_reduced = decimate(img_reduced, 2, axis = 0)
#
# img_reduced = exposure.equalize_hist(img_reduced)
# img = exposure.equalize_hist(img)

# plt.figure(1)
# plt.imshow(img, cmap=plt.get_cmap('gray'))
# plt.figure(2)
# plt.imshow(img_reduced, cmap=plt.get_cmap('gray'))
# plt.show()

#img = img_reduced

# h, w = img.shape
# X = filter_bank(img, coeff_resolution=1.0)
#
# for i in range(0, len(X)):
#     plt.figure(i)
#     features = X[:, i].reshape(h, w)
#     plt.imshow(features, cmap=plt.get_cmap('gray'))
#     plt.show()






