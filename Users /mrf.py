import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy
import math
from evaluation.segmentation_scoring import rejectOne_score
from sklearn.metrics import accuracy_score
import copy

def mrf_map(X, Y, mu, sigma, nb_class, max_map_iter, alpha, beta):
    """
        Goal:       Run the MRF_MAP_ICM process
        Input:      - X = labels
                    - Y = extracted features
                    - nb_class
                    - max_map_iter = maximum number of iteration to run
                    - alpha = weight of the unary potential function
                    - beta = weight of the pairwise potential function
        Output:     Regularized label field
    """
    im_x, im_y = Y.shape[:2]
    y = Y.reshape((-1,1))
    global_nrj = np.zeros((im_x * im_y, nb_class))
    sum_nrj_map = []
    neigh_mask = beta * np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    #neigh_mask = beta * np.asarray([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 0, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])
    #neigh_mask = beta * np.asarray([[0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

    for it in range(max_map_iter):
        unary_nrj = np.copy(global_nrj)
        pairwise_nrj = np.copy(global_nrj)

        for l in range(nb_class):
            yi = y - mu[l]
            temp1 = (yi * yi) / (2 * np.square(sigma[l])) + math.log(sigma[l])
            unary_nrj[:, l] = unary_nrj[:, l] + temp1[:, 0]

            label_mask = np.zeros(X.shape)
            label_mask[np.where(X == l)] = 1
            pairwise_nrj_temp = scipy.ndimage.convolve(label_mask, neigh_mask, mode='constant')
            pairwise_nrj_temp *= -1.0
            pairwise_nrj[:, l] = pairwise_nrj_temp.flat

        global_nrj = np.copy(alpha * unary_nrj + pairwise_nrj)
        X = np.copy(np.argmin(global_nrj, axis=1).reshape((im_x, im_y)))

        sum_nrj_map.append(np.sum(np.amin(global_nrj, axis=1).reshape((-1, 1))))
        if it >= 3 and np.std(sum_nrj_map[it-2:it])/np.absolute(sum_nrj_map[it]) < 0.01:
            break

    return X

def run_mrf(label_field, feature_field, nb_class, max_map_iter, weight):
    """
        Goal:       Run the MRF_MAP_ICM process
        Input:      - label_field = SVM outputted labels
                    - feature_field = extracted features
                    - nb_class
                    - max_map_iter = maximum number of iteration to run
                    - weight = weights
        Output:     Regularized label field
    """
    img_shape = feature_field.shape
    mu = []
    sigma = []

    blurred = gaussian_filter(feature_field, sigma=weight[2])
    y = blurred.reshape((-1,1))

    for i in range(nb_class):
        yy = y[np.where(label_field == i)[0], 0]
        mu.append(np.mean(yy))
        sigma.append(np.std(yy))

    X = label_field.reshape(img_shape)
    X = X.astype(int)
    Y = blurred

    return mrf_map(X, Y, mu, sigma, nb_class, max_map_iter, weight[0], weight[1])


def train_mrf(label_field, feature_field, nb_class, max_map_iter, weight, threshold_learning, label_true, threshold_sensitivity, threshold_error=1.0):
    """
        Goal:       Weight Learning by maximizing the pixel accuracy + sensitivity condition
        Input:      - label_field = SVM outputted labels
                    - feature_field = extracted features
                    - nb_class
                    - max_map_iter = maximum number of iteration to run
                    - weight = weight initialization
                    - threshold_learning = learning rate
                    - label_true = ground true
                    - threshold_sensitivity = condition on the sensitivity
                    - threshold_error = condition on the 'error'
        Output:     - weight[0] = weight of the unary potential function
                    - weight[1] = weight of the pairwise potential function
                    - weight[2] = standard deviation for Gaussian kernel
    """
    s = [1] * len(weight)
    d = [1] * len(weight)

    weight_init = copy.deepcopy(weight)

    res = run_mrf(label_field, feature_field, nb_class, max_map_iter, weight)
    best_score = accuracy_score(label_true, res.reshape((-1, 1)))

    while any(ss >= threshold_learning for ss in s):
        for i in range(len(weight)):
            weight_cur = copy.deepcopy(weight)
            weight_cur[i] = weight[i] + s[i] * d[i]

            res = run_mrf(label_field, feature_field, nb_class, max_map_iter, weight_cur)

            acc_mrf = accuracy_score(label_true, res.reshape((-1, 1)))
            scores = rejectOne_score(feature_field, label_true, res.reshape((-1,1)), visualization=False, show_diffusion=True, min_area = 0)

            if acc_mrf > best_score and scores[0] > threshold_sensitivity and scores[1] < threshold_error:
                best_score = acc_mrf
                weight[i] = weight_cur[i]

            else:
                s[i] = float(s[i])/2
                d[i] = -d[i]

    print 'Learned parameters: ' + str(weight)
    print 'Learned parameters: ' + str(weight_init)
    print ' \n'
    if cmp(weight_init, weight) == 0:
        print 'No parameter changes: re-examine the learning conditions'

    return weight

