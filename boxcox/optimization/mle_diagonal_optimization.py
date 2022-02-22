import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from boxcox.optimization import Optimizer
from scipy.stats import boxcox


class MLEDiagonalOptimizer(Optimizer):
    """
    Optimization of the lambda values for a given dataset with columnwise MLE for the Boxcox parameter
    """

    def __init__(self):

        self.validation_performance = 0

    def run(self, features, labels, classifier):
        """
        optimizes the lambda values
        :param features: samples to classify
        :param labels: class labels for the samples
        :param classifier: classifier to train and evaluate
        :return: optimal lambdas for every feature
        """
        classifier_temp = clone(classifier)
        features_temp = np.copy(features)
        lambdas = np.zeros(features.shape[1])

        # Transform data with MLE for lambda
        for idx, column in enumerate(features_temp.T):
            features_temp[:, idx], lambda_ = boxcox(column)
            lambdas[idx] = lambda_

        pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier_temp)])

        pipeline.fit(features_temp, labels)
        prediction = pipeline.predict(features_temp)

        self.validation_performance = accuracy_score(prediction, labels)
        # print("Validation accuracy: " + str(self.validation_performance))

        return lambdas

    def get_validation_performance(self):
        """
        :return: validation performance of hyperparameter tuning
        """
        return self.validation_performance
