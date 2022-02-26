import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from boxcox.optimization import Optimizer
from scipy.stats import boxcox, yeojohnson


class SphericalOptimization(Optimizer):
    """
    Optimization of the lambda values for a given dataset with spherical optimization (one scalar lambda for all
    columns)
    """

    def __init__(self, nr_lambdas, method='box_cox'):
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

        lambda_opt = -5
        performance_tmp = -np.inf

        for lambda_ in lambda_list:
            features_temp = np.copy(features_original)
            for idx, column in enumerate(features_temp.T):
                if self.method == 'box_cox':
                    features_temp[:, idx] = boxcox(column, lambda_)
                elif self.method == 'yeo_johnson':
                    features_temp[:, idx] = yeojohnson(column, lambda_)
                else:
                    features_temp[:, idx] = boxcox(column, lambda_)

            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier_temp)])

            pipeline.fit(features_temp, labels)
            prediction = pipeline.predict(features_temp)
            acc = accuracy_score(prediction, labels)

            if acc > performance_tmp:
                lambda_opt = lambda_
                performance_tmp = acc

        self.validation_performance = performance_tmp
        # print("Validation accuracy: " + str(self.validation_performance))

        lambdas = np.zeros(features.shape[1])
        lambdas = np.full_like(lambdas, lambda_opt)

        return lambdas

    def get_validation_performance(self):
        """
        :return: validation performance of hyperparameter tuning
        """
        return self.validation_performance
