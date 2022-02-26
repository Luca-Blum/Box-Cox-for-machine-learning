import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from boxcox.optimization import Optimizer
from scipy.stats import boxcox, yeojohnson


class DiagonalOptimization(Optimizer):
    """
    Optimization of the lambda values for a given dataset with Diagonal optimization (each column independent)
    """

    def __init__(self, nr_lambdas=11, method='box_cox'):
        """
        param nr_lambdas: number of lambdas used for each 1D-gridserch
        param method: transformation method to use ['box_cox', 'yeo_johnson']
        """

        self.nr_lambdas = nr_lambdas
        self.method = method
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
        features_original = np.copy(features)
        lambda_list = np.linspace(start=-5, stop=5, num=self.nr_lambdas)

        lambdas = np.zeros(features.shape[1])
        performance_tmp = -np.inf

        for column_idx in range(features_original.shape[1]):
            features_temp = np.copy(features_original)

            for lambda_ in lambda_list:

                if self.method == 'box_cox':
                    features_temp[:, column_idx] = boxcox(features_original[:, column_idx], lmbda=lambda_)
                elif self.method == 'yeo_johnson':
                    features_temp[:, column_idx] = yeojohnson(features_original[:, column_idx], lmbda=lambda_)
                else:
                    features_temp[:, column_idx] = boxcox(features_original[:, column_idx], lmbda=lambda_)

                pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier_temp)])

                pipeline.fit(features_temp, labels)
                prediction = pipeline.predict(features_temp)
                acc = accuracy_score(prediction, labels)

                if acc > performance_tmp:
                    lambdas[column_idx] = lambda_
                    performance_tmp = acc

        self.validation_performance = performance_tmp
        # print("Validation accuracy: " + str(self.validation_performance))

        return lambdas

    def get_validation_performance(self):
        """
        :return: validation performance of hyperparameter tuning
        """
        return self.validation_performance
