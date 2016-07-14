from sklearn.utils import shuffle
import random

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