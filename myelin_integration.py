from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
import pickle
from contour_detection import acti_contour
from skimage.filters import gaussian


######################################################################################################################
#                                                     # Import data                                                    #
#####################################################################################################################
result_number_m = 6
folder_m = 'results_myelin/result_%s'%result_number_m
myelin = pickle.load(open(folder_m+"/results.pkl", "rb"))
myelin_pred_proba = myelin['img_pred_proba']
myelin_pred = myelin['img_pred']
print myelin_pred.shape[0]*myelin_pred.shape[1]
img = myelin['img']

result_number_a = 2
folder_a = 'resultsPipeline/result_%s'%result_number_a
axon = pickle.load(open(folder_a+"/results.pkl", "rb"))
axon_pred = axon['y_pred_mrf']
print axon_pred.shape


print '--Postprocessing Active Contour\n'

im_contour, mask = acti_contour(myelin_pred, axon_pred, param_contour={'alpha': 0.015, 'beta': 100})
y_pred_contour = im_contour.reshape(-1,1)


fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(gaussian(img, 0.5))
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.hold(True)
plt.imshow(im_contour, alpha=0.7)
plt.title('active contour')

