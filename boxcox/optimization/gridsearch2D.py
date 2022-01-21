import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from boxcox.optimization import Optimizer
from scipy.stats import boxcox


class Gridsearch2D(Optimizer):
    """
    Optimization of the lambda values for a given 2D dataset and classifier with a gridsearch
    """

    def __init__(self, nr_points=11, lower_bound=-5, upper_bound=5):
        """
        :param nr_points: number of points evenly space on grid
        :param lower_bound: lower bound of grid
        :param upper_bound: upper bound of grid
        """

        self.nr_points = nr_points
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.validation_performance = 0

    def run(self, features, labels, classifier):
        """
        optimizes the lambda values
        :param features: samples to classify
        :param labels: class labels for the samples
        :param classifier: classifier to train and evaluate
        :return: optimal lambdas for every feature
        """

        lambda1 = np.linspace(start=self.lower_bound, stop=self.upper_bound, num=self.nr_points)
        lambda2 = np.linspace(start=self.lower_bound, stop=self.upper_bound, num=self.nr_points)

        lambdas = []
        performance_tmp = 0

        for l1_idx, l1 in enumerate(lambda1):
            for l2_idx, l2 in enumerate(lambda2):

                classifier_temp = clone(classifier)
                features_temp = np.copy(features)

                # Transform data
                features_temp[:, 0] = boxcox(features_temp[:, 0], l1)
                features_temp[:, 1] = boxcox(features_temp[:, 1], l2)

                pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier_temp)])

                pipeline.fit(features_temp, labels)
                prediction = pipeline.predict(features_temp)
                acc = accuracy_score(prediction, labels)

                if acc > performance_tmp:
                    lambdas = [l1, l2]
                    performance_tmp = acc

        self.validation_performance = performance_tmp
        # print("Validation accuracy: " + str(self.validation_performance))

        return lambdas

    def get_validation_performance(self):
        """
        :return: validation performance of hyperparameter tuning
        """
        return self.validation_performance
