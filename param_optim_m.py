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
import pandas as pd
from sampling import sampling
from sklearn import cross_validation
import warnings
warnings.filterwarnings('ignore')


data = pickle.load(open("data/groundTruth.pkl", "rb"))
img = data['image']
img = exposure.equalize_hist(img)
mask = data['mask_m']

h, w = img.shape
test_size = 0.90

#######################################################################################################################
#                                            Building training and test sets                                          #
#######################################################################################################################

img_train = img[:, :w*(1-test_size)]
mask_train = mask[:, :w*(1-test_size)]
img_test = img[:, w*(1-test_size):]
mask_test = mask[:, w*(1-test_size):]

h_test, w_test = img_test.shape
h_train, w_train = img_train.shape

X_train = features(img_train, 3)
X_test = features(img_test, 3)

y_train = np.ravel(mask_train.reshape(-1, 1))
y_test = np.ravel(mask_test.reshape(-1, 1))

#######################################################################################################################
#                                            Building training and test sets                                          #
#######################################################################################################################

n_train_s = 30000
X_train_s, y_train_s = sampling(X_train, y_train, n_train_s, balanced=False)

df = pd.DataFrame(columns=('max_depth', 'mean', 'std'))

h, dim = X_train_s.shape
param = [5,7, 9, 10, 15, 20, 22]

i = 0
for variable in param:
   clf = Classifier_myelin(pc=False, max_depth=variable)
   scores = cross_validation.cross_val_score(estimator=clf, X=X_train_s, y=y_train_s, cv=4, n_jobs=4, scoring='accuracy')
   df.loc[i] = [variable, scores.mean(), scores.std()]
   i += 1

print '\n Grid Search Results'
df_sort = df.sort(columns=['mean', 'std'], ascending=True)
print df_sort
print'---------------------'

df.plot(x='max_depth', y='mean')
plt.fill_between(df['max_depth'], df['mean']-2*df['std'], df['mean']+2*df['std'], color='b', alpha=0.2)
plt.show()



