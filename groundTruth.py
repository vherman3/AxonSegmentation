import numpy as np
from sklearn import preprocessing
from PIL import Image
import matplotlib.pyplot as plt
import pickle

file = 'data/'

img = Image.open(file+'image.png').convert('L')
img = np.array(img)
h, w = img.shape

im = Image.open(file+'initial_mask.tif').convert('L')
mask_init = preprocessing.binarize(np.array(im), threshold=100)

im = Image.open(file+'rectified_mask2.png').convert('L')
mask_add = preprocessing.binarize(np.array(im), threshold=100)

im = Image.open(file+'suppress_mask.tif').convert('L')
mask_supp = preprocessing.binarize(np.array(im), threshold=100)

mask = mask_add + (1-mask_supp) * mask_init

data = {}
data['image'] = img
data['mask'] = mask

with open(file+'data2.pkl', 'wb') as handle:
    pickle.dump(data, handle)


fig1 = plt.figure()
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(mask, alpha=0.7)

plt.show()

