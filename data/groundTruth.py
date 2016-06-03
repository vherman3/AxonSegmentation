import numpy as np
from sklearn import preprocessing
from PIL import Image
import pickle


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

data = {}
data['image'] = img
data['mask'] = gt
with open('groundTruth.pkl', 'wb') as handle:
    pickle.dump(data, handle)

#######################################################################################################################
#                                                      Visualization                                                  #
#######################################################################################################################
#import matplotlib.pyplot as plt

# fig1 = plt.figure()
# plt.imshow(img, cmap=plt.get_cmap('gray'))
# plt.hold(True)
# plt.imshow(gt, alpha=0.7)
# plt.show()

