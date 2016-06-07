
import pickle
import numpy as np
from skimage import exposure
from features_extraction import features
from sampling import sampling
from sklearn.externals import joblib
import classifier
from mrf import train_mrf
from mrf import run_mrf
from segmentation_scoring import rejectOne_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tabulate import tabulate
import matplotlib.pyplot as plt
import time
from pylab import savefig
import os
from multipred import parallized_pred


result_number = 5
folder = 'resultsPipeline/result_%s'%result_number
if not os.path.exists(folder):
    os.makedirs(folder)


text = '----------REPORT----------'

#######################################################################################################################
#                                                      Import data                                                    #
#######################################################################################################################

print '--Importing data \n'

data = pickle.load(open("data/groundTruth.pkl", "rb"))
img = data['image']
img = exposure.equalize_hist(img)
mask = data['mask']

#######################################################################################################################
#                                            Building training and test sets                                          #
#######################################################################################################################

print '--Building sets \n'

h, w = img.shape
test_size = 0.80
text+= '\n\n---Parameters--'
text+= '\n-test_size = %s' %test_size

img_train = img[:h*(1-test_size), :]
mask_train = mask[:h*(1-test_size), :]

img_test = img[h*(1-test_size):, :]
mask_test = mask[h*(1-test_size):, :]

img_test = img_test[:-200, :-400]
mask_test = mask_test[:-200, :-400]

X_train = features(img_train, 3)
X_test = features(img_test, 3)

y_train = np.ravel(mask_train.reshape(-1, 1))
y_test = np.ravel(mask_test.reshape(-1, 1))

text+= '\n-train_size = (%s pixels)'% y_train.shape[0]

#MRF parameters
nb_class = 2
max_map_iter = 10
alpha = 1.0
beta = 1.0
sigma_blur = 1.0
threshold_learning = 0.1
threshold_sensitivity = 0.65
threshold_error = 0.10

# #-------Samples of the training set to reducte computational time-------#
# n_train_s = 65000
# X_train_s, y_train_s = sampling(X_train, y_train, n_train_s, balanced=False)
# text+= '\n-sampled train = True (%s samples)'%n_train_s
# #----------------

#print 'classRatio =  ', float(len(y_train[y_train == 1]))/len(y_train[y_train == 0])

#######################################################################################################################
#                                            Train and predict                                                        #
#######################################################################################################################

print '--Training \n'

clf = classifier.Classifier(verbose=True)

start = time.clock()
#clf.fit(X_train_s, y_train_s)
clf.fit(X_train, y_train)
weight = train_mrf(y_pred_svm_train, img_train, nb_class, max_map_iter, [alpha, beta, sigma_blur], threshold_learning, y_train, threshold_sensitivity)
#weight = train_mrf(y_pred_svm_train, img_train, nb_class, max_map_iter, [alpha, beta, sigma_blur], threshold_learning, y_train, threshold_sensitivity, threshold_error)
training_time = time.clock()-start

print '--Predicting \n'

start = time.clock()
y_pred = parallized_pred(X_test, clf)
y_pred_train = parallized_pred(X_train, clf)
prediction_time = time.clock()-start

results = {}
results['img']= img_test
results['img_train']= img_train
results['y_pred']= y_pred
results['y_true']= y_test
results['y_train']= y_train
results['y_pred_train']= y_pred_train

text+= '\n\n---Computational time--'
text+= '\n-training_time = %s s' % (training_time)
text+= '\n-prediction_time = %s s' % (prediction_time)

######################################################################################################################
#                                            Postprocessing  - MRF                                                    #
#######################################################################################################################
print '--Postprocessing \n'

img_mrf = run_mrf(y_pred_svm_test, img_test, nb_class, max_map_iter, weight)
img_mrf = img_mrf == 1
y_pred_mrf = img_mrf.reshape(-1, 1)

results['y_pred_mrf'] = y_pred_mrf
results['y_pred_mrf'] = y_pred_mrf

#######################################################################################################################
#                                           Scoring                                                                   #
#######################################################################################################################
print '--Evaluating performances \n'

acc_train = accuracy_score(y_train, y_pred_train)
pre_train = precision_score(y_train, y_pred_train,average=None)

acc_test = accuracy_score(y_test, y_pred)
pre_test = precision_score(y_test, y_pred, average=None)

acc_mrf = accuracy_score(y_test, y_pred_mrf)
pre_mrf = precision_score(y_test, y_pred_mrf, average=None)

score_test = rejectOne_score(img_test, y_test, y_pred, visualization=False, show_diffusion=True, min_area=0)
score_train = rejectOne_score(img_train, y_train, y_pred_train, visualization=False, show_diffusion=True, min_area=0)
score_mrf = rejectOne_score(img_test, y_test, y_pred_mrf, visualization=False, show_diffusion=True, min_area=0)

headers = ["set", "accuracy","sensitivity", "errors", "diffusion"]
table = [["test", acc_test, score_train[0], score_train[1], score_train[2]],
        ["train", acc_train, score_test[0], score_test[1], score_test[2]],
        ["mrf test", acc_mrf, score_mrf[0], score_mrf[1], score_mrf[2]]]

subtitle2 = '\n\n---Scores---\n'
scores = tabulate(table, headers)
text = text+subtitle2+scores

#######################################################################################################################
#                                           Visualization                                                             #
#######################################################################################################################


plt.figure(1)
plt.title('Prediction on test')
plt.imshow(img_test, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(y_pred.reshape(img_test.shape[0], img_test.shape[1]), alpha=0.7)
savefig(folder+'/test_prediction'+'.png')


plt.figure(2)
plt.title('Prediction on train')
plt.imshow(img_train, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(y_pred_train.reshape(img_train.shape[0], img_train.shape[1]), alpha=0.7)
savefig(folder+'/train_prediction'+'.png')

plt.figure(3)
plt.title('Prediction on test with mrf')
plt.imshow(img_test, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(img_mrf, alpha=0.7)
savefig(folder+'/test_prediction_mrf'+'.png')

#######################################################################################################################
#                                            Saving                                                                   #
#######################################################################################################################
print '\n--Saving results \n'

with open(folder+'/results'+'.pkl', 'wb') as handle:
     pickle.dump(results, handle)

f = open(folder+'/report'+'.txt', 'w')
f.write(text)
f.close()

folder_clf = 'resultsPipeline/result_%s/classifier/'%result_number
if not os.path.exists(folder_clf):
    os.makedirs(folder_clf)

joblib.dump(clf, folder_clf+'clf'+'.pkl')
