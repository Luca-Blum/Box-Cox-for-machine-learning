import numpy as np
import scipy.stats
from sklearn.pipeline import Pipeline
from sklearn import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from boxcox.optimization import Optimizer


class IterativeOptimizer(Optimizer):
    """
    Iteratively optimizes the lambdas to get the highest performance
    This reduces the optimization complexity to n 1D optimizations problems instead of one global pD optimization
    problem where p is the number of features/columns.
    """
    def __init__(self, nr_lambdas=21, init='box_cox', method='box_cox', epochs=16, shuffle_epoch=2, shift_epoch=8,
                 finer_epoch=4, seed=42):
        """
        :param init: method for the initial transformation
        :param method: method for individual transformation. If 'random' then
        :param nr_lambdas:
        :param epochs: number of rounds used for optimizing the complete dataset
        :param shuffle_epoch: number of rounds until the order of iterative optimization gets reshuffled
        :param shift_epoch: number of rounds until lambda gets perturb to escape local minima
        :param finer_epoch: number of round until the gridsearch gets finer
        :param seed: random seed
        """
        self.init_method = init
        self.performance_history = None
        self.validation_performance = 0
        self.nr_lambdas = nr_lambdas
        self.init = init
        self.method = method
        self.epochs = epochs
        self.shuffle_epoch = shuffle_epoch
        self.shift_epoch = shift_epoch
        self.finer_epoch = finer_epoch
        self.seed = seed

    def run(self, features, labels, classifier):
        """
        First it transforms the features according to the given method. Afterwards it optimizes one column after another
        while holding the other features constant. The number of rounds specifies how many times this optimization
        scheme is applied to the complete dataset.
        :param features: data samples
        :param labels: class labels of samples
        box_cox is used for the individual feature transformation
        :param classifier: classifier to apply
        :return:
        """

        if self.finer_epoch > self.shift_epoch:
            self.finer_epoch = self.shift_epoch

        features_original = np.copy(features)

        # optimize
        performance_tmp = -np.inf

        lambda_list = np.linspace(start=-5, stop=5, num=self.nr_lambdas)

        self.performance_history = np.zeros(self.epochs * features.shape[1])

        indices = np.arange(features_original.shape[1])
        rng = np.random.default_rng(seed=self.seed)

        if self.init in ['box_cox', 'yeo_johnson']:
            lambdas = np.zeros(features.shape[1])
        else:
            lambdas = np.random.random(features.shape[1])

        features_transformed = np.copy(features)

        for idx, column in enumerate(features_transformed.T):
            if self.init == 'box_cox':
                features_transformed[:, idx], max_log = scipy.stats.boxcox(column)
                lambdas[idx] = max_log
            elif self.init == 'yeo_johnson':
                features_transformed[:, idx], max_log = scipy.stats.yeojohnson(column)
                lambdas[idx] = max_log
            else:
                features_transformed[:, idx] = scipy.stats.boxcox(column, lambdas[idx])

        finer_counter = 0
        start_round = 1
        
        initial_point = np.copy(lambdas)
        shift = False

        for round_ in range(self.epochs):
            if round_ != 0:
                finer_counter += 1
                if round_ % self.shift_epoch == 0:
                    finer_counter = 0
                    initial_point = np.random.random(features_transformed.shape[1])
                    for idx, column in enumerate(features_transformed.T):
                        if self.method == 'box_cox':
                            features_transformed[:, idx] = scipy.stats.boxcox(features_original[:, idx],
                                                                              initial_point[idx])
                        elif self.method == 'yeo_johnson':
                            features_transformed[:, idx] = scipy.stats.yeojohnson(features_original[:, idx],
                                                                                  initial_point[idx])
                        else:
                            features_transformed[:, idx] = scipy.stats.boxcox(features_original[:, idx],
                                                                              initial_point[idx])

                    lambda_list = np.linspace(start=-5, stop=5, num=self.nr_lambdas)
                    start_round = 1
                    shift = True

                elif finer_counter % self.finer_epoch == 0:
                    start_round = 0
                    lambda_list = 0.5 * lambda_list

                if round_ % self.shuffle_epoch == 0:
                    rng.shuffle(indices)

            for iteration in range(features_transformed.shape[1]):

                features_tmp = np.copy(features_transformed)

                idx = indices[iteration]

                lambda_finer = lambda_list + (1 - start_round) * lambdas[idx]

                for lambda_ in lambda_finer:
                    if self.method == 'box_cox':
                        features_tmp[:, idx] = scipy.stats.boxcox(features_original[:, idx], lmbda=lambda_)
                    elif self.method == 'yeo_johnson':
                        features_tmp[:, idx] = scipy.stats.yeojohnson(features_original[:, idx], lmbda=lambda_)
                    else:
                        features_tmp[:, idx] = scipy.stats.boxcox(features_original[:, idx], lmbda=lambda_)

                    classifier_temp = clone(classifier)
                    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier_temp)])

                    pipeline.fit(features_tmp, labels)
                    prediction = pipeline.predict(features_tmp)
                    acc = accuracy_score(prediction, labels)

                    if acc > performance_tmp:
                        if shift:
                            lambdas = np.copy(initial_point)
                            shift = False
                        lambdas[idx] = lambda_
                        performance_tmp = acc

                if self.method == 'box_cox':
                    features_transformed[:, idx] = scipy.stats.boxcox(features_original[:, idx], lambdas[idx])
                elif self.method == 'yeo_johnson':
                    features_transformed[:, idx] = scipy.stats.yeojohnson(features_original[:, idx], lambdas[idx])
                else:
                    features_transformed[:, idx] = scipy.stats.boxcox(features_original[:, idx], lambdas[idx])

                self.performance_history[round_ * features.shape[1] + iteration] = performance_tmp

        # self.validation_performance = performance_tmp
        print("Validation accuracy: " + str(performance_tmp))
        
        return lambdas

    def get_validation_performance(self):
        """
        :return: validation performance of hyperparameter tuning
        """
        return self.validation_performance
