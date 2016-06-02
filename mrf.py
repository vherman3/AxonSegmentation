import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy
from skimage import color
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import math
from time import time
from skimage import feature

#######################################################################################################################
#                                                      MRF-MAP                                                        #
#######################################################################################################################
def mrf_map(X, Y, mu, sigma, nb_class, map_iter, beta, verbose):
    im_x, im_y = Y.shape[:2]
    y = Y.reshape((-1,1))
    U = np.zeros((im_x * im_y, nb_class))
    sum_U_MAP = np.zeros((1, map_iter))

    for it in range(map_iter):
        U1 = np.copy(U)
        U2 = np.copy(U)

        for l in range(nb_class):
            yi = y - mu[l, 0]
            temp1 = (yi * yi) / (2 * np.square(sigma[l,0])) + math.log(sigma[l, 0])
            U1[:,l] = U1[:,l] + temp1[:, 0]

            idx = 0
            for i in range(im_x):
                for j in range(im_y):
                    uu2 = 0

                    if i > 0 and l != X[i-1, j]:
                        uu2 += beta[0]
                    if i < im_x-1 and l != X[i+1, j]:
                        uu2 += beta[0]
                    if j > 0 and l != X[i, j-1]:
                        uu2 += beta[1]
                    if j < im_y-1 and l != X[i, j+1]:
                        uu2 += beta[1]


                    U2[idx, l] = uu2
                    idx += 1

        U = np.copy(U1 + U2)

        temp = np.amin(U, axis=1).reshape((-1, 1))
        X = np.copy(np.argmin(U, axis=1).reshape((im_x, im_y)))
        sum_U_MAP[0, it] = np.sum(temp)

        if it >= 3 and np.std(sum_U_MAP[0, it-2:it])/sum_U_MAP[0, it] < 0.001:
            break

    sum_U = 0
    x = X.reshape((-1,1))
    for i in range(im_x*im_y):
        sum_U += U[i, x[i, 0]]

    if verbose == 1:
        plt.figure()
        plt.plot(range(map_iter), sum_U_MAP[0, :map_iter], 'ro-')
        plt.xlabel('MAP iteration')
        plt.ylabel('Sum of U MAP')
        plt.title('Sum of U MAP')
        plt.grid(True)
        plt.show()

    return X, sum_U

#######################################################################################################################
#                                                      HMRF-EM                                                        #
#######################################################################################################################
def hmrf_em(X, Y, mu, sigma, nb_class, em_iter, map_iter, beta, verbose):
    im_x, im_y = Y.shape[:2]
    y = Y.reshape((-1, 1))
    prob = np.zeros((im_x * im_y, nb_class))
    sum_u = np.zeros((1, em_iter))

    for it in range(em_iter):

        res_mrf_map = mrf_map(X, Y, mu, sigma, nb_class, map_iter, beta, 0)

        X = np.copy(res_mrf_map[0])
        sum_u[0, it] = res_mrf_map[1]

        for l in range(nb_class):
            temp1 = np.exp(-np.square(y - mu[l, 0]) / (2 * np.square(sigma[l, 0]))) / math.sqrt(2 * math.pi * np.square(sigma[l, 0]))
            temp2 = np.zeros((im_x*im_y, 1))

            idx = 0
            for i in range(im_x):
                for j in range(im_y):
                    u = 0

                    if i > 0 and l != X[i-1, j]:
                        u += beta[0]
                    if i < im_x-1 and l != X[i+1, j]:
                        u += beta[0]
                    if j > 0 and l != X[i, j-1]:
                        u += beta[1]
                    if j < im_y-1 and l != X[i, j+1]:
                        u += beta[1]

                    temp2[idx, 0] = u
                    idx += 1

            prob[:, l] = (temp1*np.exp(-temp2))[:, 0]

        temp3 = (np.sum(prob, axis=1)).reshape((-1,1))
        prob = prob/temp3

        for l in range(nb_class):
            mu[l, 0] = np.dot(np.transpose(prob[:, l]),y)
            mu[l, 0] = mu[l, 0]/np.sum(prob[:, l])
            sigma[l, 0] = np.dot(np.transpose(prob[:, l]), np.square(y - mu[l,0]))
            sigma[l, 0] = sigma[l, 0]/np.sum(prob[:, l])
            sigma[l, 0] = np.sqrt(sigma[l, 0])

        if it >= 3 and np.std(sum_u[0, it - 2 : it])/sum_u[0, it] < 0.001:
            break


    if verbose == 1:
        plt.figure()
        plt.plot(range(em_iter), sum_u[0,:em_iter], 'ro-')
        plt.xlabel('EM iteration')
        plt.ylabel('Sum of U')
        plt.title('Sum of U in each EM iteration')
        plt.grid(True)
        plt.show()

    return X, mu, sigma

#######################################################################################################################
#                                                      Score Evaluation                                               #
#######################################################################################################################

def compute_sensitivity_specificity(confusion_matrix, nb_class):
    sensitivity = np.zeros((nb_class, 1))
    specificity = np.zeros((nb_class, 1))

    for i in range(nb_class):
        sensitivity[i, 0] = float(confusion_matrix[i, i])/(np.sum(confusion_matrix[:, i]))
        specificity[i, 0] = float(np.trace(confusion_matrix) - confusion_matrix[i, i])/(np.sum(np.sum(confusion_matrix))-np.sum(confusion_matrix[:, i]))

    return sensitivity, specificity

def compute_dice(target, estim, nb_class):
    dice = np.zeros((nb_class,1))

    for i in range(nb_class):
        where_true = target == i
        where_estim = estim == i
        dice[i,0] = float(2*np.sum(where_true == where_estim))/(where_true.shape[0]+where_estim.shape[0])

    return dice

def similarity_metrics(target, estim, nb_class, lab_names):
    headers = ["sensitiv.", "speicif.", "DICE", "precision", "recall", "f1", "support"]
    last_line_heading = 'avg / total'
    width = max(len(cn) for cn in lab_names)
    width = max(width, len(last_line_heading))
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(target, estim)
    matrix = confusion_matrix(target, estim)
    matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    sstvt, spcfct = compute_sensitivity_specificity(matrix, nb_class)
    dice = compute_dice(target, estim, nb_class)

    for i in range(nb_class):
        values = [lab_names[i]]

        for v in (sstvt[i, 0], spcfct[i, 0], dice[i,0], p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, 2)]

        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)
    report += '\n'

    values = [last_line_heading]
    for v in (np.average(sstvt.flat, weights=s),
              np.average(spcfct.flat, weights=s),
              np.average(dice.flat, weights=s),
              np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.{1}f}".format(v, 2)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)

    return report, matrix_norm

#######################################################################################################################
#                                                     Function                                                        #
#######################################################################################################################
def mrf_hmrf(axon_dict, type ='mrf',beta =[2,2]):

    nb_class = 2
    em_iter = 2
    map_iter = 5
    beta = [2, 2]
#######################################################################################################################
#                                                      Import Data                                                    #
#######################################################################################################################
    img_test = axon_dict['img']
    img_shape = img_test.shape
    ground_truth = axon_dict['y_true']
    svm_predict = axon_dict['y_pred']

#######################################################################################################################
#                                                      Preprocess                                                     #
#######################################################################################################################

    blurred = gaussian_filter(img_test, sigma=1.2)

    mu = np.zeros((nb_class,1))
    sigma = np.zeros((nb_class,1))

    y = blurred.reshape((-1,1))

    for i in range(nb_class):
        yy = y[np.where(svm_predict == i)[0], 0]
        mu[i, 0] = np.mean(yy)
        sigma[i, 0] = np.std(yy)

    X = svm_predict.reshape(img_shape)
    X = X.astype(int)
    Y = blurred

#######################################################################################################################
#                                                      Run MRF-MAP                                                    #
#######################################################################################################################

    if type == 'mrf':
        res = mrf_map(X, Y, mu, sigma, nb_class, map_iter, beta, 0)
        binary = res[0] == 1

    else :
        res = hmrf_em(X, Y, mu, sigma, nb_class, em_iter, map_iter, beta, 0)
        binary = res[0] == 1

    return binary


