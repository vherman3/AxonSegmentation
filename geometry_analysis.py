from skimage import measure
from skimage.measure import regionprops
import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import pandas as pd
from math import pi
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import stats
from mrf import mrf_hmrf
from sklearn.decomposition import PCA


def axon_mask(coords, img, transparent=False, display = True):

    m = np.zeros((img.shape[0], img.shape[1]))
    m[coords[:, 0], coords[:, 1]] = 1
    if display == True :

        if transparent:
            plt.figure(1)
            plt.imshow(img, cmap=plt.get_cmap('gray'))
            plt.hold(True)
            plt.imshow(m, alpha=0.7)
            plt.show()

        else:
            plt.figure(1)
            plt.imshow(m, cmap=plt.get_cmap('gray'))
            plt.show()

    return m

#######################################################################################################################
#                                                      Import Data                                                    #
#######################################################################################################################

file2 = 'test/'
results = pickle.load(open(file2+"results.pkl", "rb"))
img = results['img']
h, w = img.shape
y_true = results['y_true']
y_pred = results['y_pred']
img_pred = y_pred.reshape(h,w)

#######################################################################################################################
#                                                      Apply Postprocessing                                           #
#######################################################################################################################

img_mrf = mrf_hmrf(results, type='mrf')
y_pred_mrf = img_mrf.reshape(-1, 1)

img_hmrf = mrf_hmrf(results, type='hmrf')
y_pred_hmrf = img_hmrf.reshape(-1, 1)


#######################################################################################################################
#                                                      Extract features                                               #
#######################################################################################################################

labels = measure.label(img_mrf)
regions = regionprops(labels, intensity_image=img)
features = ['coordinates', 'solidity', 'area', 'circularity', 'ellipticity', 'mean_intensity', 'std_intensity']

df = pd.DataFrame(columns = features)
h, w = img.shape

i=0
for x in regions:

    a = x.area
    if a == 1:
        continue

    p = x.perimeter
    cir = float(4*pi*a)/(p**2)
    if p==0:
        continue

    max = x.major_axis_length
    min = x.minor_axis_length
    i_mean = x.mean_intensity
    i_std = np.std(x.intensity_image)

    df.loc[i] = [x.coords, x.solidity, a, cir, min/max, i_mean, i_std]
    i += 1

#######################################################################################################################
#                                                      Clustering                                                     #
#######################################################################################################################

X = df.drop('coordinates', axis=1)
features.remove('coordinates')
print features

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#pca = PCA(n_components=4)
#pca.fit(X)
#X = pca.transform(X)
#print '----- Clustering : \n PCA components : ', pca.components

outliers_fraction = 0.03

clf = OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.01)
clf.fit(X)

y_pred = clf.decision_function(X).ravel()
threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)
y_pred = y_pred > threshold
df['anomaly'] = y_pred

df_anormal = df[df['anomaly'] == False]

#######################################################################################################################
#                                                      Visualization                                                  #
#######################################################################################################################

M = np.zeros((img.shape[0], img.shape[1]))
for idx in df_anormal.index:
    m = axon_mask(df_anormal.ix[idx]['coordinates'], img, display=False)
    M += m

plt.figure(3)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(M, alpha=0.7)

X_normal = X[y_pred]
X_anormal = X[y_pred == False]


axes = (0,1)
plt.figure(4)
plt.subplot(2, 2, 1)
nor = plt.scatter(X_normal[:, axes[0]], X_normal[:, axes[1]], c='g')
anor = plt.scatter(X_anormal[:, axes[0]], X_anormal[:, axes[1]], c='r')
plt.legend((nor, anor), ('Normal Data', 'Geometric outliers'))
plt.xlabel(features[axes[0]])
plt.ylabel(features[axes[1]])

axes = (0,3)
plt.subplot(2, 2, 2)
nor = plt.scatter(X_normal[:, axes[0]], X_normal[:, axes[1]], c='g')
anor = plt.scatter(X_anormal[:, axes[0]], X_anormal[:, axes[1]], c='r')
plt.legend((nor, anor), ('Normal Data', 'Geometric outliers'))
plt.xlabel(features[axes[0]])
plt.ylabel(features[axes[1]])

axes = (2,4)
plt.subplot(2, 2, 3)
nor = plt.scatter(X_normal[:, axes[0]], X_normal[:, axes[1]], c='g')
anor = plt.scatter(X_anormal[:, axes[0]], X_anormal[:, axes[1]], c='r')
plt.legend((nor, anor), ('Normal Data', 'Geometric outliers'))
plt.xlabel(features[axes[0]])
plt.ylabel(features[axes[1]])


axes = (3,4)
plt.subplot(2, 2, 4)
nor = plt.scatter(X_normal[:, axes[0]], X_normal[:, axes[1]], c='g')
anor = plt.scatter(X_anormal[:, axes[0]], X_anormal[:, axes[1]], c='r')
plt.legend((nor, anor), ('Normal Data', 'Geometric outliers'))
plt.xlabel(features[axes[0]])
plt.ylabel(features[axes[1]])


plt.show()






