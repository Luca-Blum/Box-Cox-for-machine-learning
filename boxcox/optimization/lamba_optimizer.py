from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract base class for the optimizers of lambda
    """
    @abstractmethod
    def run(self, features, labels, classifier):
        """
        optimizes the lambdas
        :param features: samples to classify
        :param labels: class labels for the samples
        :param classifier: classifier to train and evaluate
        :return: optimal lambdas for every feature
        """
        pass
