import pickle
import numpy as np
import classifier
from features_extraction import features
from skimage import exposure
from sklearn.metrics import accuracy_score
import random
from sklearn.utils import shuffle
import time
from Segmentation_scoring import segment_score, rejectOne_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import cross_validation
import pandas as pd
from sklearn import feature_selection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mrf import mrf_hmrf


data = pickle.load(open("data/data.pkl", "rb"))
img = data['image']
img = exposure.equalize_hist(img)

mask = data['mask']

X = features(img, 3)
y = mask.reshape(-1, 1)

h, w = img.shape
test_size = 0.87

img_train = img[:h*(1-test_size), :]
img_test = img[h*(1-test_size):, :]

mask_train = mask[:h*(1-test_size), :]
mask_test = mask[h*(1-test_size):, :]

X_train = features(img_train, 3)
X_test = features(img_test, 3)

y_train = np.ravel(mask_train.reshape(-1, 1))
y_test = np.ravel(mask_test.reshape(-1, 1))


#######################################################################################################################
#                                                      Extract samples                                                #
#######################################################################################################################

def sampling(X, y, n_samples, balanced=True):

    if balanced == True :

        X_1 = random.sample(X[y == 1], n_samples/2)
        X_0 = random.sample(X[y == 0], n_samples/2)

        y_1 = np.ones((n_samples/2, 1))
        y_0 = np.zeros((n_samples/2, 1))

        y_sample = np.concatenate((y_1, y_0), axis=0)
        X_sample = np.concatenate((X_1, X_0), axis=0)

        X_sample, y_sample = shuffle(X_sample, y_sample)

    if balanced == False :
        X_sample, y_sample = shuffle(X, y)
        X_sample = X_sample[:n_samples, :]
        y_sample = y_sample[:n_samples]


    return [X_sample, y_sample]


#######################################################################################################################
#                                                      Features evaluation                                             #
#######################################################################################################################
# n_train_s = 50000
# n_test_s = 5000
#
# for i in range(4):
#     X_train_s, y_train_s = sampling(X_train, y_train, n_train_s, balanced=False)
#     scaler = StandardScaler()
#     scaler.fit(X_train_s)
#     scaler.transform(X_train_s)
#     chi2, pval = feature_selection.chi2(X_train_s, y_train_s)
#     fea_ind = np.argsort(pval)

#######################################################################################################################
#                                                      Grid Search                                                    #
#######################################################################################################################
# n_train_s = 20000
# n_test_s = 5000
# X_train_s, y_train_s = sampling(X_train, y_train, n_train_s, balanced=False)
# X_test_s, y_test_s = sampling(X_test, y_test, n_test_s, balanced=False)
#
#
# df = pd.DataFrame(columns=('C', 'n_components', 'n_features', 'mean', 'std'))
#
# h, dim = X_train_s.shape
#
# C_ = [0.01, 0.1, 1, 1.5, 3]
# n_components_ = [30, 35, 38, 40, 45, 50]
# n_features_ = [1, 10, 20, 30, 40, 70, 80, dim]
#
# #np.random.seed(123)
# C = 1.5
# #n_comt
# n_features = n_features_[-1]
#
# i = 0
# for n_components in n_components_:
#    clf = classifier.Classifier(pc=True, C=C, n_components=n_components)
#    scores = cross_validation.cross_val_score(estimator=clf, X=X_train_s[:, fea_ind[:n_features]], y=y_train_s, cv=4, n_jobs=4, scoring='accuracy')
#    df.loc[i] = [C, n_components, n_features, scores.mean(), scores.std()]
#    i += 1
#
# print '\n Grid Search Results'
# df_sort = df.sort(columns=['mean', 'std'], ascending=True)
# print df_sort
# print'---------------------'
#
# df.plot(x='n_components', y='mean')
# plt.fill_between(df['n_components'], df['mean']-2*df['std'], df['mean']+2*df['std'], color='b', alpha=0.2)
# plt.show()

#######################################################################################################################
#                                                      Train on samples                                               #
#######################################################################################################################

n_train_s = 10000
n_test_s = 10000
X_train_s, y_train_s = sampling(X_train, y_train, n_train_s, balanced=False)
X_test_s, y_test_s = sampling(X_test, y_test, n_test_s, balanced=False)


print 'classRatio =  ', float(len(y_train[y_train == 1]))/len(y_train[y_train == 0])

clf = classifier.Classifier(C=1.5, n_components=35, verbose=True, class_weight=None)
print 'Training start'
start = time.clock()
clf.fit(X_train_s, y_train_s)
print 'time on train clf: ', time.clock()-start, 's'

print '\nPrediction start'
y_pred_s = clf.predict(X_test_s)
y_pred_train = clf.predict(X_train_s)

start = time.clock()
y_pred_test = clf.predict(X_test)
print 'time on test clf: ', time.clock()-start, 's'


results = {}
results['img']= img_test
results['y_pred']= y_pred_test
results['y_true']= y_test
#
# with open('test/result_2.pkl', 'wb') as handle:
#     pickle.dump(results, handle)

#######################################################################################################################
#                                                      Print some scores                                              #
#######################################################################################################################

print '\n==========Accuracy========='
print 'predict sampled test :', accuracy_score(y_test_s, y_pred_s), 'precision :', precision_score(y_test_s, y_pred_s, average=None)
print 'predict sampled train :', accuracy_score(y_train_s, y_pred_train), 'precision :', precision_score(y_train_s, y_pred_train, average=None)
print 'predict on test :', accuracy_score(y_test, y_pred_test), 'precision :', precision_score(y_test, y_pred_test, average=None)

print '\n==========Segmentation Score========='
sensitivity_test, errors_test = segment_score(img_test, y_test, y_pred_test, visualization=False, min_area=0)
print 'predict on test :', sensitivity_test, errors_test

print '\n==========RejectOne Score========='
sensitivity_test, errors_test = rejectOne_score(img_test, y_test, y_pred_test, visualization=True, min_area=0)
print 'predict on test :', sensitivity_test, errors_test


