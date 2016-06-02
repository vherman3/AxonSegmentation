from PIL import Image
import numpy as np
from sklearn import preprocessing
import scipy
import pickle
from Segmentation_scoring import segment_score, rejectOne_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from mrf import mrf_hmrf


# Script to compare basic method used by the AxonSegmentation Toolbox, Machine Learning methods without postprocessing#
# and Machine Learning with postprocessing (markov random fields)
#######################################################################################################################

file = 'Comparaison/basic_method/'
img = Image.open(file+'Step_2.jpg').convert('L')
img_basic = preprocessing.binarize(np.array(img), threshold=100)

h, w = img_basic.shape
test_size = 0.87
img_basic = img_basic[h*(1-test_size):, :]
y_pred_basic = img_basic.reshape(-1, 1)

file2 = 'results/'
results = pickle.load(open(file2+"results.pkl", "rb"))
img = results['img']
h, w = img.shape
y_true = results['y_true']
y_pred = results['y_pred']

#######################################################################################################################
#                                                      Apply Postprocessing                                           #
#######################################################################################################################

img_mrf = mrf_hmrf(results, type='mrf')
y_pred_mrf = img_mrf.reshape(-1, 1)

img_hmrf = mrf_hmrf(results, type='hmrf')
y_pred_hmrf = img_hmrf.reshape(-1, 1)


#for beta_i in (2, 6):
    #y_pred_mrf = img_mrf.reshape(-1, 1)
    #sensitivity_mrf, errors_mrf, diffusion_mrf = rejectOne_score(img, y_true, y_pred_mrf)

#######################################################################################################################
#                                                      Cropping                                                       #
#######################################################################################################################

def crop(data, coeff, h, w, type = 'vector'):
    if type =='image':
        data_cropped = data[coeff:-coeff, coeff:-coeff]
    if type =='vector':
        data = data.reshape(h,w)
        data_cropped = data[coeff:-coeff, coeff:-coeff].reshape(-1, 1)
    return data_cropped

coeff = 30 # We do not analyse side axons
img = crop(img, coeff, h, w, type='image')
y_true = crop(y_true, coeff, h, w)
y_pred = crop(y_pred, coeff, h, w)
y_pred_basic = crop(y_pred_basic, coeff, h, w)
y_pred_mrf = crop(y_pred_mrf, coeff, h, w)
y_pred_hmrf = crop(y_pred_hmrf, coeff, h, w)

#######################################################################################################################
#                                                      Results                                                       #
#######################################################################################################################

sensitivity_bas, errors_bas, diffusion_bas = rejectOne_score(img, y_true, y_pred_basic, visualization=True, min_area=0, show_diffusion=True)
sensitivity, errors, diffusion = rejectOne_score(img, y_true, y_pred, visualization=False, min_area=0, show_diffusion=True)
sensitivity_mrf, errors_mrf, diffusion_mrf = rejectOne_score(img, y_true, y_pred_mrf, visualization=False, min_area=0, show_diffusion= True)
sensitivity_hmrf, errors_hmrf, diffusion_hmrf = rejectOne_score(img, y_true, y_pred_hmrf, visualization=False, min_area=0, show_diffusion= True)

print '\n========Segmentation precision========='
print 'basic method || sensitivity :', sensitivity_bas, ', errors : ', errors_bas,  ', diffusion coeff : ', diffusion_bas
print 'ML method || ', sensitivity, errors, diffusion
print 'ML method + MRF || ', sensitivity_mrf, errors_mrf, diffusion_mrf
print 'ML method + HMRF || ', sensitivity_hmrf, errors_hmrf, diffusion_hmrf

print '\n========Pixel precision================'
print 'basic || precision :', precision_score(y_true, y_pred_basic, average=None)[:2]
print 'ML ', precision_score(y_true, y_pred, average=None)[:2]
print 'ML method + MRF ', precision_score(y_true, y_pred_mrf, average=None)[:2]
print 'ML method + HMRF ', precision_score(y_true, y_pred_hmrf, average=None)[:2]

print '\n=====Pixel accuracy===================='
print 'basic method || ', round(accuracy_score(y_true, y_pred_basic), 3)
print 'ML method || ', round(accuracy_score(y_true, y_pred), 3)
print 'ML method + MRF || ', round(accuracy_score(y_true, y_pred_mrf), 3)
print 'ML method + HMRF || ', round(accuracy_score(y_true, y_pred_hmrf), 3)

