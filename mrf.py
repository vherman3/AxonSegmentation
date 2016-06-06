import numpy as np
from scipy.ndimage.filters import gaussian_filter
import math
from sklearn.metrics import accuracy_score
from Segmentation_scoring import rejectOne_score

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
    U = np.zeros((im_x * im_y, nb_class))
    sum_U_MAP = np.zeros((1, max_map_iter))

    for it in range(max_map_iter):
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
                        uu2 += beta
                    if i < im_x-1 and l != X[i+1, j]:
                        uu2 += beta
                    if j > 0 and l != X[i, j-1]:
                        uu2 += beta
                    if j < im_y-1 and l != X[i, j+1]:
                        uu2 += beta

                    U2[idx, l] = uu2
                    idx += 1

        U = np.copy(alpha*U1 + U2)

        X = np.copy(np.argmin(U, axis=1).reshape((im_x, im_y)))

        sum_U_MAP[0, it] = np.sum(np.amin(U, axis=1).reshape((-1, 1)))
        if it >= 3 and np.std(sum_U_MAP[0, it-2:it])/np.absolute(sum_U_MAP[0, it]) < 0.01:
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
    mu = np.zeros((nb_class,1))
    sigma = np.zeros((nb_class,1))

    blurred = gaussian_filter(feature_field, sigma=weight[2])
    y = blurred.reshape((-1,1))

    for i in range(nb_class):
        yy = y[np.where(label_field == i)[0], 0]
        mu[i, 0] = np.mean(yy)
        sigma[i, 0] = np.std(yy)

    X = label_field.reshape(img_shape)
    X = X.astype(int)
    Y = blurred

    return mrf_map(X, Y, mu, sigma, nb_class, max_map_iter, weight[0], weight[1])

def train_mrf(label_field, feature_field, nb_class, max_map_iter, weight, threshold_learning, label_true, threshold_sensitivity):
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

        Output:     - weight[0] = weight of the unary potential function
                    - weight[1] = weight of the pairwise potential function
                    - weight[2] = standard deviation for Gaussian kernel
    """
    s = np.ones((len(weight), 1))
    d = np.ones((len(weight), 1))

    weight_init = weight
    alpha = weight[0]
    beta = weight[1]
    sigma_blur = weight[2]

    res = run_mrf(label_field, feature_field, nb_class, max_map_iter, weight)
    best_score = accuracy_score(label_true, res.reshape((-1, 1)))

    while any(ss >= threshold_learning for ss in s):
        for i in range(len(weight)):

            if i == 0:
                alpha = weight[i] + s[i, 0] * d[i, 0]
            elif i == 1:
                beta = weight[i] + s[i, 0] * d[i, 0]
            else:
                sigma_blur = weight[i] + s[i, 0] * d[i, 0]

            res = run_mrf(label_field, feature_field, nb_class, max_map_iter, [alpha, beta, sigma_blur])

            acc_mrf = accuracy_score(label_true, res.reshape((-1, 1)))
            sensitiv = rejectOne_score(feature_field, label_true, res.reshape((-1,1)), visualization=False, show_diffusion=True, min_area = 0)[0]

            if acc_mrf > best_score and sensitiv > threshold_sensitivity:
                best_score = acc_mrf

                if i == 0:
                    weight[i] = alpha
                elif i == 1:
                    weight[i] = beta
                else:
                    weight[i] = sigma_blur

            else:
                s[i, 0] = float(s[i, 0])/2
                d[i, 0] = -d[i, 0]

    if np.array_equal(weight_init, weight):
        print 'No parameter changes: re-examine the learning conditions'

    return weight