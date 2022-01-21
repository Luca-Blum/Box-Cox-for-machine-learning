import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y


class ClassifierOptimization(BaseEstimator, ClassifierMixin):
    """
    Fit own approach into sklearn API
    """
    def __init__(self, classifier, optimizer=None):
        """
        :param classifier: classifier to train
        :param optimizer: which optimzer to use to get the optimal lambda values
        """
        self.classifier = classifier
        self.lambdas = []
        self.optimizer = optimizer
        self.scaler = None
        self.minMaxScaler = None

    def fit(self, data, targets):
        """
        Training the classifier with optimization of lambda
        :param data: features/samples
        :param targets: class labels
        :return: trained classifier
        """

        data, targets = check_X_y(data, targets)

        features = np.copy(data)
        labels = np.copy(targets)

        self.minMaxScaler = MinMaxScaler(feature_range=(1, 2))
        features = self.minMaxScaler.fit_transform(features, labels)

        self.lambdas = self.optimizer.run(features, labels, self.classifier)

        for idx, column in enumerate(features.T):
            features[:, idx] = scipy.stats.boxcox(column, self.lambdas[idx])

        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        self.classifier.fit(features, labels)

        return self

    def predict(self, data):
        """
        Predict class label for given data
        :param data: sample to make predictions on
        :return: prediction
        """

        # check_is_fitted(self.classifier)

        features = np.copy(data)

        features = self.minMaxScaler.transform(features)

        for idx, column in enumerate(features.T):
            features[:, idx] = scipy.stats.boxcox(column, lmbda=self.lambdas[idx])

        features = self.scaler.transform(features)

        return self.classifier.predict(features)

    def get_lambdas(self):
        """
        :return: trained/optimized lambda values
        """

        return self.lambdas
