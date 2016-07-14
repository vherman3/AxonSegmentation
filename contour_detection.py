import pickle
import matplotlib.pyplot as plt
from skimage.measure import regionprops,label
import numpy as np
from skimage.segmentation import active_contour
import pandas as pd
from skimage.filters import gaussian
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import find_contours
from skimage.draw import polygon
from sklearn.metrics import accuracy_score
from segmentation_scoring import dice
from functools import partial
from multiprocessing import Pool
import warnings
import time
warnings.filterwarnings('ignore')


#######################################################################################################################
#                                                      Active contour                                                 #
#######################################################################################################################

def acti_contour_object(region, img, param_contour):

    alpha = param_contour['alpha']
    beta = param_contour['beta']

    mask = np.zeros((img.shape[0],img.shape[1]))

    if region.area > 100:
        mask[region.coords[:, 0], region.coords[:, 1]] = 1
        filled = binary_fill_holes(mask).astype(int)
        contour = find_contours(filled, 0.5)[0]

    # else :
    #     r = np.sqrt(x.area/np.pi)
    #     s = np.linspace(0, 2*np.pi, 40)
    #     x_1 = x.centroid[1] + r*np.cos(s)
    #     x_2 = x.centroid[1] + r*np.sin(s)
    #     contour = np.array([x_1, x_2]).T

        init_contour = np.array([contour[:, 1], contour[:, 0]]).T
        snake = active_contour(gaussian(img, 0.1), init_contour, alpha=alpha, beta=beta)
        rr, cc = polygon(snake[:, 1], snake[:, 0])## !!! attention le snake peut depasser de l'image...

    try:
        mask[rr, cc] = 1
    except:
        pass
    return mask


def acti_contour(img, im_pred, param_contour={'alpha': 0.015, 'beta': 20}, sampling = False, n_sample = 100):


    mask = np.zeros((im_pred.shape[0], im_pred.shape[1]))

    labels_pred = label(im_pred)
    regions_pred = regionprops(labels_pred)

    if sampling == True :
        regions_pred = np.random.choice(regions_pred, size=n_sample)


    mapfunc = partial(acti_contour_object, img = im_pred, param_contour=param_contour)

    # masks = []
    # for region in regions_pred :
    #     mask = acti_contour_object(region, param_contour)
    #     masks.append(mask)

    pool = Pool(processes=3)
    masks = pool.map(mapfunc, regions_pred)
    #masks = map(mapfunc, regions_pred)
    pool.close()
    pool.join()

    cleaned = [x for x in masks if not x is None]
    cleaned = np.asarray(cleaned)
    mask = sum(cleaned)

    result = im_pred + mask
    result[result > 1] = 1
    return [result, mask]


######################################################################################################################
#                                                     # Import data                                                  #
#####################################################################################################################
#
# result_number = 3
# folder = 'resultsPipeline/result_%s'%result_number
# data = pickle.load(open(folder+"/results.pkl", "rb"))
#
# y_pred = data['y_pred_mrf']
# img = data['img']
# y_true = data['y_true']
#
# h,w = img.shape
# im_pred = y_pred.reshape((h, w))
# im_true = y_true.reshape((h,w))
#
# df = dice(img,y_true,y_pred)
# df = df[df['area']>100]
# print 'mean', df['dice'].mean(axis=0)
# print 'std', df['dice'].std(axis=0)
#
# #######################################################################################################################
# #                                                      Searching best parameters                                      #
# #######################################################################################################################
#
# results = pd.DataFrame(columns=('beta', 'alpha', 'mean dice', 'std dice'))
# beta_ = [0.1]#150, 200, 1000]
# alpha_ = [0.005, 0.01, 0.1, 1000, 10000, 100000] #30, 40]
# alpha = 0.015
#
# start = time.time()
#
# i = 0
# for alpha in alpha_:
#     for beta in beta_:
#         param_contour = {'alpha': alpha, 'beta': beta}
#         im_pred2, contour_filled = acti_contour(img, im_pred, param_contour=param_contour)
#         y_pred2 = im_pred2.reshape(-1, 1)
#         D = dice(im_pred2, y_true, y_pred2)
#         D = D[D['area'] >= 100]
#         results.loc[i] = [beta, alpha, D['dice'].mean(axis=0), D['dice'].std(axis=0)]
#         i += 1
#         print 'iteration : %s/%s' %(i, len(beta_)*len(alpha_))
#         print 'beta =',beta, 'alpha =', alpha
#
#         fig = plt.figure(i)
#         ax = fig.add_subplot(111)
#         plt.gray()
#         ax.imshow(gaussian(img, 0.5))
#         ax.set_xticks([]), ax.set_yticks([])
#         ax.axis([0, img.shape[1], img.shape[0], 0])
#         plt.hold(True)
#         plt.imshow(contour_filled, alpha=0.7)
#         plt.title('active contour')
#
# print 'optimization time', time.time() - start
# plt.show()
#
#
# print '\n Grid Search Results'
# print results
# results_sort = results.sort(columns=['mean dice', 'std dice'], ascending=True)
# print results_sort
# print'---------------------'
#
# variable = 'beta'
# results.plot(x=variable, y='mean dice')
# #plt.fill_between(results[variable], results['mean']-2*results['std'], results['mean']+2*results['std'], color='b', alpha=0.2)
# plt.show()


#######################################################################################################################
#                                                      Visualization                                                  #
#######################################################################################################################

# im_pred2, mask = acti_contour(img, im_pred, param_contour={'alpha': 0.015, 'beta': 100})
# y_pred2 = im_pred2.reshape(-1, 1)
#
# print 'score :', accuracy_score(y_pred2, y_true)
#
# fig = plt.figure(1)
# ax = fig.add_subplot(111)
# plt.gray()
# ax.imshow(gaussian(img, 0.5))
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
# plt.hold(True)
# plt.imshow(im_pred2, alpha=0.7)
# plt.title('active contour')
#
# fig = plt.figure(2)
# ax = fig.add_subplot(111)
# plt.gray()
# ax.imshow(gaussian(img, 0.5))
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
# plt.hold(True)
# plt.imshow(im_pred, alpha=0.7)
# plt.title('mrf')
#
#
# fig = plt.figure(3)
# ax = fig.add_subplot(111)
# plt.gray()
# ax.imshow(gaussian(img, 0.5))
# ax.set_xticks([]), ax.set_yticks([])
# ax.axis([0, img.shape[1], img.shape[0], 0])
# plt.hold(True)
# plt.imshow(im_true, alpha=0.7)
# plt.title('groundTruth')
#
#
# plt.show()
