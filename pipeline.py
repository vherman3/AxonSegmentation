
import pickle
import numpy as np
from skimage import exposure
from features_extraction import features
from sampling import sampling
from sklearn.externals import joblib
import classifier
from mrf import mrf_hmrf
from segmentation_scoring import rejectOne_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from visualization import visualize
import time


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
test_size = 0.99
text+= '\n\n---Parameters--'
text+= '\n\n-test_size = %s' % (test_size)

img_train = img[:h*(1-test_size), :]
img_test = img[h*(1-test_size):, :]


mask_train = mask[:h*(1-test_size), :]
mask_test = mask[h*(1-test_size):, :]

X_train = features(img_train, 3)
X_test = features(img_test, 3)

y_train = np.ravel(mask_train.reshape(-1, 1))
y_test = np.ravel(mask_test.reshape(-1, 1))

# #-------Samples of the training set to reducte computational time-------#
# n_train_s = 50000
# X_train_s, y_train_s = sampling(X_train, y_train, n_train_s, balanced=False)
# #----------------

#print 'classRatio =  ', float(len(y_train[y_train == 1]))/len(y_train[y_train == 0])

#######################################################################################################################
#                                            Train and predict                                                        #
#######################################################################################################################

print '--Training \n'

clf = classifier.Classifier()

start = time.clock()
#clf.fit(X_train_s, y_train_s)
clf.fit(X_train, y_train)
training_time = time.clock()-start

print '--Predicting \n'

start = time.clock()
y_pred = clf.predict(X_test)
y_pred_train = clf.predict(X_train)
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

img_mrf = mrf_hmrf(results, type='mrf')
y_pred_mrf = img_mrf.reshape(-1, 1)

results['y_pred_mrf'] = y_pred_mrf
results['y_pred_mrf'] = y_pred_mrf

#######################################################################################################################
#                                           Scoring and visualization                                                 #
#######################################################################################################################
print '--Evaluating performances \n'

acc_train = accuracy_score(y_train, y_pred_train)
pre_train = precision_score(y_train, y_pred_train,average=None)
acc_test = accuracy_score(y_test, y_pred)
pre_test = precision_score(y_test, y_pred, average=None)
acc_mrf = accuracy_score(y_test, y_pred)
pre_mrf = precision_score(y_test, y_pred, average=None)

score_test = rejectOne_score(img_test, y_test, y_pred, visualization=False, show_diffusion=True, min_area=0)
score_train = rejectOne_score(img_train, y_train, y_pred_train, visualization=False, show_diffusion=True, min_area=0)
score_mrf = rejectOne_score(img_test, y_test, y_pred_mrf, visualization=False, show_diffusion=True, min_area=0)

headers = ["set", "accuracy","sensitivity", "errors", "diffusion"]
table = [["train", acc_test, score_train[0], score_train[1], score_train[2]],
        ["train", acc_train, score_test[0], score_test[1], score_test[2]],
        ["mrf test", acc_mrf, score_mrf[0], score_mrf[1], score_mrf[2]]]

subtitle2 = '\n---Scores---\n\n'
scores = tabulate(table, headers)

#######################################################################################################################
#                                            Saving                                                                   #
#######################################################################################################################
print '\n--Saving results \n'

file_number = '_1'
folder = 'example/resultsPipeline/'
with open(folder+'results'+file_number+'.pkl', 'wb') as handle:
     pickle.dump(results, handle)
joblib.dump(clf, folder+'classifiers/'+'clf_'+file_number+'.pkl')

text = text+subtitle2+scores

f = open(folder+'scores'+file_number+'.txt', 'w')
f.write(text)
f.close()

