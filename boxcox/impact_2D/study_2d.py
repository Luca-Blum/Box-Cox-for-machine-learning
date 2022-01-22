import pathlib
import pickle

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns


class Study2D:
    """
    Understand the influence of the Box-Cox transformation for 2D datasets
    """
    def __init__(self, name=None, lower_bound=-5, upper_bound=5):
        """
        :param name: prefix used to store results (string)
        :param lower_bound: lower bound for the lambda parameter
        :param upper_bound: upper bound for the lambda parameter
        """
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, features, labels, method='box_cox', nr_lambdas=11, nr_features=2,
                 seed=42, data_path=None):
        """
        :param features: samples
        :param labels: class labels
        :param method: transformation method
        :param nr_lambdas: number of lambdas in each direction
        :param nr_features: number of columns
        :param seed: seed for random number generators to make result reproducible
        :param data_path: path to store data
        :return:
        """

        pipelines = {"NN": Pipeline(steps=[('scaler', StandardScaler()),
                                           ('classifier', MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000,
                                                                        random_state=seed))]),
                     "linear": Pipeline(steps=[('scaler', StandardScaler()),
                                               ('classifier',
                                                SGDClassifier(loss="perceptron", random_state=seed))]),
                     "knn": Pipeline(steps=[('scaler', StandardScaler()),
                                            ('classifier', KNeighborsClassifier())]),
                     "bayesian": Pipeline(steps=[('scaler', StandardScaler()),
                                                 ('classifier', GaussianNB())]),
                     "SVM": Pipeline(steps=[('scaler', StandardScaler()),
                                            ('classifier', SVC(random_state=seed))])}

        lambda1 = np.linspace(start=self.lower_bound, stop=self.upper_bound, num=nr_lambdas)
        lambda2 = np.linspace(start=self.lower_bound, stop=self.upper_bound, num=nr_lambdas)

        performance = np.zeros((nr_lambdas, nr_lambdas, len(pipelines)))
        stds = np.zeros((nr_lambdas, nr_lambdas, len(pipelines)))

        gauss = np.zeros((nr_lambdas, nr_lambdas, nr_features, 2))
        skew = np.zeros((nr_lambdas, nr_lambdas, nr_features))

        features_temp = np.copy(features)
        performance_base, stds_base = self.evaluate_base(features_temp, labels, pipelines, seed)

        gauss_base = self.evaluate_gaussianity(features)
        skew_base = self.evaluate_skewness(features)

        description = "number of features " + str(nr_features) + "\nnumber of lambdas: \t\t\t\t" + \
                      str(nr_lambdas**2) + "\nfeature transformation method: \t" + method

        for l1_idx, l1 in enumerate(lambda1):
            for l2_idx, l2 in enumerate(lambda2):
                features_temp = np.copy(features)

                if method == 'box_cox':
                    features_temp[:, 0] = scipy.stats.boxcox(features_temp[:, 0], lmbda=l1)
                    features_temp[:, 1] = scipy.stats.boxcox(features_temp[:, 1], lmbda=l2)
                elif method == 'yeo_johnson':
                    features_temp[:, 0] = scipy.stats.yeojohnson(features_temp[:, 0], lmbda=l1)
                    features_temp[:, 1] = scipy.stats.yeojohnson(features_temp[:, 1], lmbda=l2)
                else:
                    features_temp[:, 0] = scipy.stats.boxcox(features_temp[:, 0], lmbda=l1)
                    features_temp[:, 1] = scipy.stats.boxcox(features_temp[:, 1], lmbda=l2)

                gauss[l1_idx, l2_idx] = self.evaluate_gaussianity(features_temp)
                skew[l1_idx, l2_idx] = self.evaluate_skewness(features_temp)

                pipeline_idx = 0
                print()
                print('Performances for lambda = (' + str(l1) + ', ' + str(l2) + ')')

                # Reset pipelines to make results reproducible
                pipelines = {"NN": Pipeline(steps=[('scaler', StandardScaler()),
                                           ('classifier', MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000,
                                                                        random_state=seed))]),
                             "linear": Pipeline(steps=[('scaler', StandardScaler()),
                                                       ('classifier',
                                                        SGDClassifier(loss="perceptron", random_state=seed))]),
                             "knn": Pipeline(steps=[('scaler', StandardScaler()),
                                                    ('classifier', KNeighborsClassifier(n_neighbors=10))]),
                             "bayesian": Pipeline(steps=[('scaler', StandardScaler()),
                                                         ('classifier', GaussianNB())]),
                             "SVM": Pipeline(steps=[('scaler', StandardScaler()),
                                                    ('classifier', SVC(random_state=seed))])}

                for key, pipeline in pipelines.items():
                    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
                    n_scores = cross_val_score(pipeline, features_temp, labels, scoring='accuracy', cv=cv, n_jobs=-1,
                                               error_score='raise')

                    acc = np.mean(n_scores)
                    std = np.std(n_scores)
                    print('Accuracy of ' + key + ':  %.3f (%.3f)' % (float(acc), float(std)))

                    performance[l1_idx, l2_idx, pipeline_idx] = acc
                    stds[l1_idx, l2_idx, pipeline_idx] = std
                    pipeline_idx += 1

        self.save_data(performance, stds, performance_base, stds_base, lambda1, gauss, skew, gauss_base, skew_base,
                       pipelines, description, data_path)

        return performance, stds, performance_base, stds_base, lambda1, gauss, skew, gauss_base, skew_base, \
               pipelines, description

    @staticmethod
    def show_data(features, labels, name="", store_path=None):
        """
        Plot data
        :param features: dataset predictors
        :param labels: corresponding class labels
        :param name: name of dataset
        :param store_path: path to store plot
        """
        plt.figure()

        color = ['g' if label == 1 else 'y' for label in labels]
        # plot x,y data with c as the color vector, set the line width of the markers to 0
        plt.scatter(features[:, 0], features[:, 1], c=color, lw=0, marker=".")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if store_path is not None:
            pathlib.Path(store_path).mkdir(parents=True, exist_ok=True)
            plot_path = pathlib.Path.joinpath(store_path, name)
            plt.savefig(plot_path)
        else:
            plt.show()

        plt.close()

        return

    @staticmethod
    def plot_boxcox_influence_data(features, labels, performance, pipelines, lambdas, store_path=None):
        """
        Visualize the influence of lambda to the data
        :param features: columns of dataframe
        :param labels: class label
        :param performance: accuracies for the different classifiers
        :param pipelines: used pipelines for classification
        :param lambdas: used lambdas for box-cox transformation
        :param store_path: path to store the plots. If None then the plots are shown
        :return: None
        """

        for l1_idx, l1 in enumerate(lambdas):

            fig = plt.figure(figsize=(15, 9), dpi=120)
            for l2_idx, l2 in enumerate(lambdas):

                features_temp = np.copy(features)

                features_temp[:, 0] = scipy.stats.boxcox(features_temp[:, 0], lmbda=l1)
                features_temp[:, 1] = scipy.stats.boxcox(features_temp[:, 1], lmbda=l2)

                color = ['g' if label == 1 else 'y' for label in labels]

                feature1 = np.array([features_temp[idx, :] for idx, label in enumerate(labels) if label == 1])
                feature1.reshape([-1, 2])
                feature2 = np.array([features_temp[idx, :] for idx, label in enumerate(labels) if label == -1])
                feature2.reshape([-1, 2])

                mean1 = np.mean(feature1, axis=0)
                mean2 = np.mean(feature2, axis=0)

                median1 = np.median(feature1, axis=0)
                median2 = np.median(feature2, axis=0)

                # plot the data

                i = l2_idx % 3
                j = l2_idx // 3

                ax1 = plt.subplot2grid((4, 3), (j, i))

                # plot x,y data with c as the color vector, set the line width of the markers to 0
                ax1.scatter(features_temp[:, 0], features_temp[:, 1], c=color, lw=0, marker=".")
                ax1.scatter(mean1[0], mean1[1], c='b', s=10, marker="x")
                ax1.scatter(median1[0], median1[1], c='aqua', s=10, marker="P")
                ax1.scatter(mean2[0], mean2[1], c='r', s=10, marker="x")
                ax1.scatter(median2[0], median2[1], c='pink', s=10, marker="P")

                text = ""
                classifiers = list(pipelines.keys())
                for idx, c in enumerate(classifiers):
                    if idx == len(classifiers) - 1:
                        text += c[0] + " %.5f" % performance[l1_idx, l2_idx, idx]
                    else:
                        text += c[0] + " %.5f \n" % performance[l1_idx, l2_idx, idx]
                ax1.text(1.2, 0.75, text, horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes)

                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height * 0.85])

                ax1.set_title('L2 = ' + str(l2))

                plt.suptitle("Data with L1 = " + str(l1), fontsize=20)

            fig.legend(['class 1', 'class 2', 'mean 1', 'median 1', 'mean 2', 'median 2'],
                       loc='upper left', bbox_to_anchor=(0.25, 0.05), ncol=len(labels), bbox_transform=fig.transFigure)

            if store_path is not None:
                fig_path = pathlib.Path.joinpath(store_path, 'L1_' + str(int(l1)))
                plt.savefig(fig_path)
            else:
                plt.show()

            plt.close(fig)

    @staticmethod
    def plot_performance(performance, performance_base, lambdas, pipelines, store_path=None):
        """
        Plot the accuracies of the different classifiers with varying lambdas
        :param performance: accuracies (np.array(#features, #classifiers, #lambdas))
        :param performance_base: accuracies of classifiers without transformation (np.array(#features, #classifiers))
        :param lambdas: lambda values (np.array(#lambdas))
        :param pipelines: (dict)
        :param store_path: folder to store the plots if it is not None. Otherwise the plots are shown and not stored
        :return: None
        """

        for l1_idx, l1 in enumerate(lambdas):

            plt.figure(figsize=(15, 9), dpi=120)
            perf = performance[l1_idx]

            ax1 = plt.subplot(111)

            classifiers = list(pipelines.keys())
            for idx, c in enumerate(classifiers):
                ax1.plot(lambdas, perf[:, idx], label=r'$\mu$ = %.2f  ' % (np.mean(perf)) + c)
                ax1.axhline(performance_base[idx], color=ax1.get_lines()[-1].get_c(), linestyle='-.')

            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15)

            ax1.set_title('L1 = ' + str(l1))

            if store_path is not None:
                fig_path = pathlib.Path.joinpath(store_path, 'L1_' + str(int(l1)))
                plt.savefig(fig_path)
            else:
                plt.show()

            plt.close()

    @staticmethod
    def plot_histogram_one_dim(features, labels, directory, performance, pipelines, lambdas, store_path=None):
        """
        Visualize the influence of lambda to the data
        :param features: columns of dataframe
        :param labels: class label
        :param directory: indicates in which direction the histogram will be done for the multidimensional dataset (int)
        :param performance: accuracies for the different classifiers
        :param pipelines: used pipelines for classification
        :param lambdas: used lambdas for box-cox transformation
        :param store_path: path to store the plots. If None then the plots are shown
        :return: None
        """

        for l1_idx, l1 in enumerate(lambdas):

            fig = plt.figure(figsize=(15, 9), dpi=120)
            for l2_idx, l2 in enumerate(lambdas):
                features_temp = np.copy(features)

                features_temp[:, 0] = scipy.stats.boxcox(features_temp[:, 0], lmbda=l1)
                features_temp[:, 1] = scipy.stats.boxcox(features_temp[:, 1], lmbda=l2)

                feature1 = np.array([features_temp[idx, :] for idx, label in enumerate(labels) if label == 1])
                feature1.reshape([-1, 2])
                feature2 = np.array([features_temp[idx, :] for idx, label in enumerate(labels) if label == -1])
                feature2.reshape([-1, 2])

                mean1 = np.mean(feature1, axis=0)
                mean2 = np.mean(feature2, axis=0)

                median1 = np.median(feature1, axis=0)
                median2 = np.median(feature2, axis=0)

                # plot the data

                i = l2_idx % 3
                j = l2_idx // 3

                ax1 = plt.subplot2grid((4, 3), (j, i))

                df = pd.DataFrame(features_temp[:, directory].T, columns=['feature1'])
                df['labels'] = labels
                hist = sns.histplot(data=df, x='feature1', hue='labels', palette=['y', 'g'], kde=True, legend=False)
                hist.set(xlabel=None)

                plt.axvline(x=mean1[directory], c='b', linestyle='--', linewidth=1)
                plt.axvline(x=mean2[directory], c='r', linestyle='--', linewidth=1)
                plt.axvline(x=median1[directory], c='aqua', linestyle='--', linewidth=1)
                plt.axvline(x=median2[directory], c='pink', linestyle='--', linewidth=1)

                text = ""
                classifiers = list(pipelines.keys())
                for idx, c in enumerate(classifiers):
                    if idx == len(classifiers) - 1:
                        text += c[0] + " %.3f" % performance[l1_idx, l2_idx, idx]
                    else:
                        text += c[0] + " %.3f \n" % performance[l1_idx, l2_idx, idx]
                ax1.text(1.2, 0.75, text, horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes)

                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height * 0.85])

                ax1.set_title('L2 = ' + str(l2))

                plt.suptitle("Feature " + str(directory + 1) + " with L1 = " + str(l1), fontsize=20)

            fig.legend(['class 1', 'class 2', 'mean 1', 'median 1', 'mean 2', 'median 2'],
                       loc='upper left', bbox_to_anchor=(0.25, 0.05), ncol=len(labels), bbox_transform=fig.transFigure)

            if store_path is not None:
                fig_path = pathlib.Path.joinpath(store_path, 'F' + str(directory + 1) + '_L1_' + str(int(l1)))
                plt.savefig(fig_path)
            else:
                plt.show()

            plt.close(fig)

        if directory == 0:
            plt.figure(figsize=(15, 9), dpi=120)
            for l1_idx, l1 in enumerate(lambdas):

                features_temp = np.copy(features)

                features_temp[:, 0] = scipy.stats.boxcox(features_temp[:, 0], lmbda=l1)

                feature1 = np.array([features_temp[idx, :] for idx, label in enumerate(labels) if label == 1])
                feature1.reshape([-1, 2])
                feature2 = np.array([features_temp[idx, :] for idx, label in enumerate(labels) if label == -1])
                feature2.reshape([-1, 2])

                mean1 = np.mean(feature1, axis=0)
                mean2 = np.mean(feature2, axis=0)

                median1 = np.median(feature1, axis=0)
                median2 = np.median(feature2, axis=0)

                # plot the data

                i = l1_idx % 3
                j = l1_idx // 3

                ax1 = plt.subplot2grid((4, 3), (j, i))

                df = pd.DataFrame(features_temp[:, directory].T, columns=['feature1'])
                df['labels'] = labels
                hist = sns.histplot(data=df, x='feature1', hue='labels', palette=['y', 'g'], kde=True, legend=False)
                hist.set(xlabel=None)

                plt.axvline(x=mean1[directory], c='b', linestyle='--', linewidth=1)
                plt.axvline(x=mean2[directory], c='r', linestyle='--', linewidth=1)
                plt.axvline(x=median1[directory], c='aqua', linestyle='--', linewidth=1)
                plt.axvline(x=median2[directory], c='pink', linestyle='--', linewidth=1)

                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height * 0.85])

                ax1.set_title('L1 = ' + str(l1))

                plt.suptitle("Feature " + str(directory + 1) + " with every L2 ", fontsize=20)

            if store_path is not None:
                fig_path = pathlib.Path.joinpath(store_path, 'F' + str(directory + 1) + '_all')
                plt.savefig(fig_path)
            else:
                plt.show()

            plt.close()

        else:

            plt.figure(figsize=(15, 9), dpi=120)
            for l2_idx, l2 in enumerate(lambdas):
                features_temp = np.copy(features)

                features_temp[:, 1] = scipy.stats.boxcox(features_temp[:, 1], lmbda=l2)

                feature1 = np.array([features_temp[idx, :] for idx, label in enumerate(labels) if label == 1])
                feature1.reshape([-1, 2])
                feature2 = np.array([features_temp[idx, :] for idx, label in enumerate(labels) if label == -1])
                feature2.reshape([-1, 2])

                mean1 = np.mean(feature1, axis=0)
                mean2 = np.mean(feature2, axis=0)

                median1 = np.median(feature1, axis=0)
                median2 = np.median(feature2, axis=0)

                # plot the data

                i = l2_idx % 3
                j = l2_idx // 3

                ax1 = plt.subplot2grid((4, 3), (j, i))

                df = pd.DataFrame(features_temp[:, directory].T, columns=['feature1'])
                df['labels'] = labels
                hist = sns.histplot(data=df, x='feature1', hue='labels', palette=['y', 'g'], kde=True, legend=False)
                hist.set(xlabel=None)

                plt.axvline(x=mean1[directory], c='b', linestyle='--', linewidth=1)
                plt.axvline(x=mean2[directory], c='r', linestyle='--', linewidth=1)
                plt.axvline(x=median1[directory], c='aqua', linestyle='--', linewidth=1)
                plt.axvline(x=median2[directory], c='pink', linestyle='--', linewidth=1)

                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height * 0.85])

                ax1.set_title('L2 = ' + str(l2))

                plt.suptitle("Feature " + str(directory + 1) + " with every L1 ", fontsize=20)

            if store_path is not None:
                fig_path = pathlib.Path.joinpath(store_path, 'F' + str(directory + 1) + '_all')
                plt.savefig(fig_path)
            else:
                plt.show()

            plt.close()

    def plot_histograms(self, features, labels, performance, pipelines, lambdas, store_path=None):
        """
        Visualize the influence of lambda to the data
        :param features: columns of dataframe
        :param labels: class label
        :param performance: accuracies for the different classifiers
        :param pipelines: used pipelines for classification
        :param lambdas: used lambdas for box-cox transformation
        :param store_path: path to store the plots. If None then the plots are shown
        :return: None
        """
        self.plot_histogram_one_dim(features, labels, 0, performance, pipelines, lambdas, store_path)
        self.plot_histogram_one_dim(features, labels, 1, performance, pipelines, lambdas, store_path)

    @staticmethod
    def plot_heatmap_sns(performances, lambdas, pipelines, store_path=None, annotation=True):
        """
        Plots heatmap of the performances with the performance score annotated
        :param performances: performances of classifiers for different lambda values
        (np.array(#lambdas, #lambdas, #pipelines)
        :param lambdas: different lambdas used to create different performance measurements
        :param pipelines: pipelines used to create performance measurements
        :param store_path: path to store the heatmap
        :param annotation: set True if annotation should be added
        :return: None
        """
        classifiers = list(pipelines.keys())
        performances_percent = performances * 100

        for idx, classifier in enumerate(classifiers):
            plt.figure(figsize=(15, 9), dpi=120)

            fig, ax = plt.subplots()
            sns.heatmap(performances_percent[:, :, idx], annot=annotation,  fmt=".2f", annot_kws={"fontsize": 10},
                        cbar=False, cmap='viridis')

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(lambdas))+0.5)
            ax.set_yticks(np.arange(len(lambdas))+0.5)
            # ... and label them with the respective list entries
            ax.set_xticklabels([int(lmb) for lmb in lambdas], fontsize=14)
            ax.set_yticklabels([int(lmb) for lmb in lambdas], fontsize=14)

            ax.set_xlabel('Lambda 2', fontsize=14)
            ax.set_ylabel('Lambda 1', fontsize=14)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
                     rotation_mode="anchor")

            # ax.set_title("Performance for " + classifier)
            fig.tight_layout()

            if store_path is not None:
                fig_path = pathlib.Path.joinpath(store_path, classifier)
                plt.savefig(fig_path)
            else:
                plt.show()

            plt.close()

    @staticmethod
    def evaluate_base(features, labels, pipelines, seed):
        """
        Evaluates the performance of the classifiers for the original dataset without any transformation
        :param features: Features for the classifier
        :param labels: Labels of the samples
        :param pipelines: List of pipelines to optimization
        :param seed: random seed (int)

        :return: performance (np.array(#pipelines)) and standard deviations of the corresponding
        pipelines (np.array(#pipelines))
        """
        performance = np.zeros(len(pipelines))
        stds = np.zeros(len(pipelines))

        features_temp = np.copy(features)

        for pipeline_idx, (key, pipeline) in enumerate(pipelines.items()):
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
            n_scores = cross_val_score(pipeline, features_temp, labels, scoring='accuracy', cv=cv, n_jobs=-1,
                                       error_score='raise')

            acc = np.mean(n_scores)
            std = np.std(n_scores)
            print('Accuracy of ' + key + ':  %.3f (%.3f)' % (float(acc), float(std)))

            performance[pipeline_idx] = acc
            stds[pipeline_idx] = std

        return performance, stds

    @staticmethod
    def evaluate_gaussianity(features):
        """
        :param features:
        :return: gaussianity of each column (np.array(width(features))
        """

        if features.ndim < 2:
            features = features.reshape((features.shape[0], 1))

        temp = [scipy.stats.shapiro(feature) for feature in features.T]

        return np.array([[result.statistic, result.pvalue] for result in temp])

    @staticmethod
    def evaluate_skewness(features):
        """
        :param features:
        :return: skewness of each column (np.array(width(features))
        """

        if features.ndim < 2:
            features = features.reshape((features.shape[0], 1))

        return np.array([scipy.stats.skew(feature) for feature in features.T])

    def save_data(self, performance, standard_deviations, performance_base, standard_deviations_base, lambdas,
                  gauss, skew, gauss_base, skew_base, classifiers, description, path=None):
        """
        Saving the given arrays with pickle dump
        :param performance: accuracies of the pipelines (np.array)
        :param standard_deviations: sdts of the pipelines (np.array)
        :param performance_base: performance of classifier without transformation
        :param standard_deviations_base: standard deviations of classifier without transformation
        :param lambdas: lambda values (np.array)
        :param gauss: gaussianity of transformed data
        :param skew: skewness of transformed data
        :param gauss_base: gaussianity of original data
        :param skew_base: skewness of original data
        :param classifiers: List of classifiers (List)
        :param description: Text of the applied methods (String)
        :param path: path to store the data
        :return: None
        """

        if path is None:
            current_path = pathlib.Path(__file__).parent.resolve()
            data_path = pathlib.Path.joinpath(current_path, self.name)
            pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
        else:
            data_path = path

        performance_path = pathlib.Path.joinpath(data_path, 'performance.dat')
        standard_deviations_path = pathlib.Path.joinpath(data_path, 'standard_deviations.dat')
        performance_base_path = pathlib.Path.joinpath(data_path, 'performance_base.csv')
        standard_deviations_base_path = pathlib.Path.joinpath(data_path, 'standard_deviations_base.csv')
        lambdas_path = pathlib.Path.joinpath(data_path, 'lambdas.csv')
        gauss_stat_path = pathlib.Path.joinpath(data_path, 'gauss_stat.csv')
        gauss_pval_path = pathlib.Path.joinpath(data_path, 'gauss_pval.csv')
        skew_path = pathlib.Path.joinpath(data_path, 'skew.csv')
        gauss_base_path = pathlib.Path.joinpath(data_path, 'gauss_base.csv')
        skew_base_path = pathlib.Path.joinpath(data_path, 'skew_base.csv')
        classifiers_path = pathlib.Path.joinpath(data_path, 'classifiers.dat')
        description_path = pathlib.Path.joinpath(data_path, 'description.txt')

        with open(performance_path, 'wb') as f:
            pickle.dump(performance, f)
        with open(standard_deviations_path, 'wb') as f:
            pickle.dump(standard_deviations, f)
        with open(performance_base_path, 'w') as f:
            df = pd.DataFrame(performance_base)
            df.index = classifiers.keys()
            df.index.name = 'classifiers'
            df.to_csv(f, sep="\t")
        with open(standard_deviations_base_path, 'w') as f:
            df = pd.DataFrame(standard_deviations_base)
            df.index = classifiers.keys()
            df.index.name = 'classifiers'
            df.to_csv(f, sep="\t")
        with open(lambdas_path, 'w') as f:
            df = pd.DataFrame(lambdas, columns=['lambda'])
            df.index.name = 'index'
            df.to_csv(f, sep="\t")
        with open(gauss_stat_path, 'wb') as f:
            pickle.dump(gauss[:, :, :, 0], f)
        with open(gauss_pval_path, 'wb') as f:
            pickle.dump(gauss[:, :, :, 1], f)
        with open(skew_path, 'wb') as f:
            pickle.dump(skew, f)
        with open(gauss_base_path, 'w') as f:
            df = pd.DataFrame(gauss_base, columns=['statistic', 'pvalue'])
            df.index.name = 'feature'
            df.to_csv(f, sep="\t")
        with open(skew_base_path, 'w') as f:
            df = pd.DataFrame(skew_base, columns=['skewness'])
            df.index.name = 'index'
            df.to_csv(f, sep="\t")
        with open(classifiers_path, 'wb') as f:
            pickle.dump(classifiers, f)
        with open(description_path, 'w') as f:
            f.write(description)

    def load_data(self, path=None):
        """
        Load the data for a univariate analysis.
        :param path: path to data
        :return: performance (np.array), standard_deviations (np.array), lambdas (np.array),
        classifiers (List), gauss(np.array), skew(np.array), gauss_base(np.array), skew_base(np.array),
        description (String), performance_base (np.array), standard_deviations_base (np.array)
        """
        if path is None:
            current_path = pathlib.Path(__file__).parent.resolve()
            data_path = pathlib.Path.joinpath(current_path, self.name)
        else:
            data_path = path

        performance_path = pathlib.Path.joinpath(data_path, 'performance.dat')
        standard_deviations_path = pathlib.Path.joinpath(data_path, 'standard_deviations.dat')
        performance_base_path = pathlib.Path.joinpath(data_path, 'performance_base.csv')
        standard_deviations_base_path = pathlib.Path.joinpath(data_path, 'standard_deviations_base.csv')
        lambdas_path = pathlib.Path.joinpath(data_path, 'lambdas.csv')
        gauss_stat_path = pathlib.Path.joinpath(data_path, 'gauss_stat.csv')
        gauss_pval_path = pathlib.Path.joinpath(data_path, 'gauss_pval.csv')
        skew_path = pathlib.Path.joinpath(data_path, 'skew.csv')
        gauss_base_path = pathlib.Path.joinpath(data_path, 'gauss_base.csv')
        skew_base_path = pathlib.Path.joinpath(data_path, 'skew_base.csv')
        classifier_path = pathlib.Path.joinpath(data_path, 'classifiers.dat')
        description_path = pathlib.Path.joinpath(data_path, 'description.txt')

        with open(performance_path, 'rb') as f:
            performance = pickle.load(f)
        with open(standard_deviations_path, 'rb') as f:
            standard_deviations = pickle.load(f)
        with open(performance_base_path, 'r') as f:
            performance_base = pd.read_csv(f, index_col='classifiers', sep="\t").to_numpy().flatten()
        with open(standard_deviations_base_path, 'r') as f:
            standard_deviations_base = pd.read_csv(f, index_col='classifiers', sep="\t").to_numpy().flatten()
        with open(lambdas_path, 'r') as f:
            lambdas = pd.read_csv(f, index_col='index', sep="\t").to_numpy().flatten()
        with open(gauss_stat_path, 'rb') as f:
            gauss_stat = pickle.load(f)
        with open(gauss_pval_path, 'rb') as f:
            gauss_pval = pickle.load(f)
        gauss = np.stack((gauss_stat, gauss_pval))
        with open(skew_path, 'rb') as f:
            skew = pickle.load(f)
        with open(gauss_base_path, 'r') as f:
            gauss_base = pd.read_csv(f, index_col='feature', sep="\t").to_numpy()
        with open(skew_base_path, 'r') as f:
            skew_base = pd.read_csv(f, index_col='index', sep="\t").to_numpy().flatten()
        with open(classifier_path, 'rb') as f:
            classifiers = pickle.load(f)
        with open(description_path, 'r') as f:
            description = f.read()

        return performance, standard_deviations, performance_base, standard_deviations_base, lambdas, gauss, skew, \
               gauss_base, skew_base, classifiers, description
