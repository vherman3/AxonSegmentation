from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC


class Classifier(BaseEstimator):
    def __init__(self, pc=True, n_components=35, verbose=False, C=1.5, class_weight=None):

        self.n_components = n_components
        self.C = C
        self.class_weight = class_weight
        self.clf = svm.SVC(kernel='rbf', cache_size=500, C=self.C, class_weight=self.class_weight, verbose=verbose)
        self.scaler = StandardScaler()
        self.pc = pc

        if self.pc:
            self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        if self.pc:
            self.pca.fit(X)
            X = self.pca.transform(X)
        self.clf.fit(X, y)

    def predict(self, X):
        """
        # Apply the scaler to the test, using the mean and std of the train
        """
        X = self.scaler.transform(X)
        if self.pc:
            X = self.pca.transform(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        you need a method that gives class probabilities like sklearn classifiers
        """
        X = self.scaler.transform(X)
        if self.pc:
            X = self.pca.transform(X)
        return self.clf.predict_proba(X)

    def score(self):
        y_pred = self.clf.predict()


class Classifier_myelin(BaseEstimator):
    def __init__(self, pc=True, n_components=35, max_depth=10, verbose=False):

        self.max_depth = max_depth
        self.n_components = n_components
        self.clf = RFC(verbose=verbose, max_depth=self.max_depth)
        self.scaler = StandardScaler()
        self.pc = pc

        if self.pc:
            self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        if self.pc:
            self.pca.fit(X)
            X = self.pca.transform(X)
        self.clf.fit(X, y)
       #print self.clf.feature_importances_

    def predict(self, X):
        """
        # Apply the scaler to the test, using the mean and std of the train
        """
        X = self.scaler.transform(X)
        if self.pc:
            X = self.pca.transform(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        you need a method that gives class probabilities like sklearn classifiers
        """
        X = self.scaler.transform(X)
        if self.pc:
            X = self.pca.transform(X)
        return self.clf.predict_proba(X)




