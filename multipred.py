import sklearn
import pickle
import numpy as np
from functools import partial
from sklearn.externals import joblib
from features_extraction import features
import time
from skimage import exposure
import warnings
warnings.filterwarnings('ignore')


def prediction(X, clf):
    return clf.predict(X)

def easy_parallize(sequence, clf):
    from multiprocessing import Pool
    pool = Pool(processes=4)
    mapfunc = partial(prediction, clf=clf)
    result = pool.map(mapfunc, sequence)
    cleaned = [x for x in result if not x is None]
    cleaned = np.asarray(cleaned)
    cleaned = np.array([x[0] for x in cleaned])
    pool.close()
    pool.join()
    return cleaned


#######################################################################################################################
#                                            Test                                                                     #
#######################################################################################################################


result_number = 2
folder = 'resultsPipeline/result_%s'%result_number

clf = joblib.load(folder+'/classifier/clf.pkl')

data = pickle.load(open("data/groundTruth.pkl", "rb"))
img = data['image']
img = exposure.equalize_hist(img)
mask = data['mask']


X_test = features(img, 3)[:40000,:]

start = time.clock()
a = easy_parallize(X_test, clf)
para_time = time.clock()-start

print para_time

start2 = time.clock()
b = clf.predict(X_test)
norm_time = time.clock()-start2

print norm_time







