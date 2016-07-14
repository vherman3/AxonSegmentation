import sklearn
from features_extraction import features
from skimage import exposure
import matplotlib.pyplot as plt
import pickle
from classifier import Classifier_myelin
from sklearn.metrics import accuracy_score
import numpy as np
from multipred import parallized_pred
import os
from sklearn.externals import joblib
from tabulate import tabulate
import time
from pylab import savefig
import warnings
warnings.filterwarnings('ignore')

result_number = 6

comments = '\n---Comments--'
comments += 'We try random forests, max_depth=10'

folder = 'results_myelin/result_%s'%result_number
if not os.path.exists(folder):
    os.makedirs(folder)

data = pickle.load(open("data/groundTruth.pkl", "rb"))
img = data['image']
img = exposure.equalize_hist(img)
mask = data['mask_m']

h, w = img.shape
test_size = 0.80

text = '----------REPORT----------'

#######################################################################################################################
#                                            Building training and test sets                                          #
#######################################################################################################################

text+= '\n\n---Parameters--'
text+= '\n-test_size = %s' %test_size

img_train = img[:h*(1-test_size),:]
mask_train = mask[:h*(1-test_size),:]
img_test = img[h*(1-test_size):,:]
mask_test = mask[h*(1-test_size):,:]

h_test, w_test = img_test.shape
h_train, w_train = img_train.shape

X_train = features(img_train, 3)
X_test = features(img_test, 3)

y_train = np.ravel(mask_train.reshape(-1, 1))
y_test = np.ravel(mask_test.reshape(-1, 1))

classRatio = 'classRatio =  %f'%(float(len(y_train[y_train == 1]))/len(y_train[y_train == 0]))

#######################################################################################################################
#                                            Train and predict                                                        #
#######################################################################################################################

print '---fitting'
clf = Classifier_myelin(pc=0, verbose=False, max_depth=10)
start = time.time()
clf.fit(X_train, y_train)
training_time = time.time()-start

print '---prediction'
start2 = time.time()
y_pred = parallized_pred(X_test, clf)
y_pred_train = parallized_pred(X_train, clf)

y_pred_proba = clf.predict_proba(X_test)[:,1]
y_train_proba = clf.predict_proba(X_train)[:,1]
prediction_time = time.time()-start2

print prediction_time

img_pred = y_pred.reshape((h_test, w_test))
img_pred_proba = y_pred_proba.reshape((h_test, w_test))
img_pred_train = y_pred_train.reshape((h_train, w_train))


text+= '\n\n---Computational time--'
text+= '\n-training_time = %s s' % (training_time)
text+= '\n-prediction_time = %s s' % (prediction_time)


results = {}
results['img']= img_test
results['img_train']= img_train
results['y_pred']= y_pred
results['y_true']= y_test
results['y_train']= y_train
results['y_pred_train']= y_pred_train
results['img_pred_proba']= img_pred_proba
results['y_train_proba']= y_train_proba
results['img_pred']= img_pred


with open(folder+'/results'+'.pkl', 'wb') as handle:
     pickle.dump(results, handle)


#######################################################################################################################
#                                           Scoring & visualization                                                   #
#######################################################################################################################
print '---scoring'

score_test = accuracy_score(y_pred, y_test)
score_train = accuracy_score(y_pred_train, y_train)

plt.figure(1)
plt.title('Prediction on test')
plt.imshow(img_test, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(img_pred, alpha=0.7)
savefig(folder+'/predict_test'+'.png')

plt.figure(2)
plt.title('Prediction on train')
plt.imshow(img_train, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(img_pred_train, alpha=0.7)
savefig(folder+'/predict_train'+'.png')

plt.figure(3)
plt.title('Proba prediction on train')
plt.imshow(img_pred_proba, cmap=plt.get_cmap('gray'))
savefig(folder+'/predict_proba_test'+'.png')


subtitle2 = '\n\n---Scores---\n'
text+=subtitle2
text+=classRatio
text+='\nAccuracy on test : %s'%score_test
text+='\nAccuracy on train: %s'%score_train

text+=comments

#######################################################################################################################
#                                           Saving                                                                    #
#######################################################################################################################

folder_clf = 'results_myelin/result_%s/classifier/'%result_number
if not os.path.exists(folder_clf):
    os.makedirs(folder_clf)
joblib.dump(clf, folder_clf+'clf'+'.pkl')

f = open(folder+'/report'+'.txt', 'w')
f.write(text)
f.close()