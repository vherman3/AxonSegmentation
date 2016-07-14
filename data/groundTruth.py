import numpy as np
from sklearn import preprocessing
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from scipy.misc import imread

#######################################################################################################################
#                                                     Axons                                                           #
#######################################################################################################################

img = Image.open('image.png').convert('L')
img = np.array(img) # Image associated to the Ground Truth
h, w = img.shape

im = Image.open('initial_mask.tif').convert('L')
mask_init = preprocessing.binarize(np.array(im), threshold=100) # Initial axon mask built with AxonSegmentation Toolbox

im = Image.open('add_mask.png').convert('L')
mask_add = preprocessing.binarize(np.array(im), threshold=100) # Added true positives in the initial mask

im = Image.open('remove_mask.tif').convert('L')
mask_rem = preprocessing.binarize(np.array(im), threshold=100) # Removed False positive in the initial mask

gt = mask_add + (1-mask_rem) * mask_init # GroundTruth
gt[gt > 1] = 1

#######################################################################################################################
#                                                     Myelin                                                           #
#######################################################################################################################

#im_m = Image.open('initial_mask_m.tif').convert('RGB')
im_m = imread('myelins_masked.TIFF')
mask_init_m = preprocessing.binarize(np.array(im_m), threshold=100)  #Initial myelin mask built with AxonSegmentation Toolbox

im_m2 = Image.open('myelin_add.tif').convert('L')
mask_add_m = preprocessing.binarize(np.array(im_m2), threshold=100) #Added true positives myelinin the initial mask

mask_add_m = mask_add_m - gt
mask_add_m[mask_add_m != 1] = 0

gt_m = mask_add_m + mask_init_m
gt_m[gt_m > 1] = 1

data = {}
data['image'] = img
data['mask'] = gt
data['mask_m'] = gt_m
with open('groundTruth.pkl', 'wb') as handle:
    pickle.dump(data, handle)

#######################################################################################################################
#                                                      Visualization                                                  #
#######################################################################################################################

# fig1 = plt.figure()
# plt.imshow(img, cmap=plt.get_cmap('gray'))
# plt.hold(True)
# plt.imshow(gt_m, alpha=0.7)
# plt.show()

