import pathlib
import argparse
import random
import re
import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt
from pandas import read_csv
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from boxcox.optimization import Gridsearch2D
from boxcox.optimization import ClassifierOptimization
from boxcox.optimization import IterativeOptimizer


def show_data(features, labels, store_path=None):
    plt.figure()

    color = ['g' if label == 1 else 'y' for label in labels]
    # plot x,y data with c as the color vector, set the line width of the markers to 0
    plt.scatter(features[:, 0], features[:, 1], c=color, lw=0, marker=".")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if store_path is not None:
        plt.savefig(store_path)
    else:
        plt.show()

    plt.close()

    return


def evaluate_performance(features, labels, seed=42, optimizer=None):
    """
    Evaluates performance for 5 different classifiers
    :param features: samples
    :param labels: labels of samples
    :param seed: seed for random number generator
    :param optimizer: procedure to find optimized lambda parameters for Box-Cox transformation
    :return: performance of box cox transformaed data, not transformed data and the corresponding standard deviations
    """
    pipelines = {"linear": Pipeline(steps=[('classifier', SGDClassifier(loss="perceptron", random_state=seed))]),
                 "knn": Pipeline(steps=[('classifier', KNeighborsClassifier())]),
                 "bayesian": Pipeline(steps=[('classifier', GaussianNB())]),
                 "SVM": Pipeline(steps=[('classifier', SVC(random_state=seed))]),
                 "NN": Pipeline(steps=[('classifier', MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000,
                                                                    random_state=seed))])
                 }

    performance_base = np.zeros(len(pipelines))
    standard_dev_base = np.zeros(len(pipelines))

    performance_box_cox = np.zeros(len(pipelines))
    standard_dev_box_cox = np.zeros(len(pipelines))

    for idx, (key, classifier) in enumerate(pipelines.items()):
        acc, std, acc_base, std_base = evaluate_classifier_performance(features, labels, classifier,
                                                                       optimizer=optimizer)

        print('Test accuracy for ' + key + ':  %.3f (%.3f)' % (float(acc), float(std)))
        print('Test accuracy for base ' + key + ':  %.3f (%.3f)' % (float(acc_base), float(std_base)))

        performance_box_cox[idx] = acc
        standard_dev_box_cox[idx] = std
        performance_base[idx] = acc_base
        standard_dev_base[idx] = std_base

        print()

    return performance_box_cox, standard_dev_box_cox, performance_base, standard_dev_base


def evaluate_classifier_performance(features, labels, classifier, seed=42, outer_cv=10, n_repeats=5, optimizer=None):
    """
    Evaluates the performance of the model with CV and optimizing for lambda
    :param features: samples
    :param labels: class labels
    :param classifier: sklearn classifier
    :param seed: seed for random number generator
    :param outer_cv: number of folds for outer cross validation
    :param n_repeats: number of repeated cross validation runs
    :param optimizer: optimization procedure to get optimized lambda values
    :return: performance of box cox transformaed data, not transformed data and the corresponding standard deviations
    """
    classifier_temp = clone(classifier)
    pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier_temp)])

    features_temp = np.copy(features)
    cv = RepeatedStratifiedKFold(n_splits=outer_cv, n_repeats=n_repeats, random_state=seed)
    n_scores = cross_val_score(pipeline, features_temp, labels, scoring='accuracy', cv=cv, n_jobs=-1,
                               error_score='raise')

    acc = np.mean(n_scores)
    std = np.std(n_scores) / np.sqrt(len(n_scores))

    print('Test accuracy for base:  %.5f (%.5f)' % (float(acc), float(std)))

    performance_base = acc
    std_base = std

    classifier_temp = clone(classifier)

    if optimizer is None:
        pipeline = Pipeline(steps=[('optimize_lambda', ClassifierOptimization(classifier_temp, IterativeOptimizer()))])
    else:
        pipeline = Pipeline(steps=[('optimize_lambda', ClassifierOptimization(classifier_temp, optimizer))])

    cv = RepeatedStratifiedKFold(n_splits=outer_cv, n_repeats=n_repeats, random_state=seed)
    n_scores = cross_val_score(pipeline, features_temp, labels, scoring='accuracy', cv=cv, n_jobs=-1,
                               error_score='raise')
    print(n_scores)
    acc = np.mean(n_scores)
    std = np.std(n_scores) / np.sqrt(len(n_scores))
    print('Test accuracy:  %.5f (%.5f)' % (float(acc), float(std)))

    performance_box_cox = acc
    std_box_cox = std

    return performance_box_cox, std_box_cox, performance_base, std_base


def run_sonar(optimizer, grid):
    random.seed(42)
    np.random.seed(42)

    # Sonar data set
    current_path = pathlib.Path(__file__).parent.resolve()
    data_path = pathlib.Path.joinpath(current_path, 'data_sonar')
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

    sonar_path = pathlib.Path.joinpath(current_path, 'sonar.csv')
    dataset = read_csv(sonar_path, header=0)
    data = dataset.values

    # separate into input and output columns
    X, y = data[:, :-1], data[:, -1]
    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))

    y = 2 * y - 1

    if not grid:
        print("Full \n")
        print(X)

        perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y,
                                                                                                optimizer=optimizer)

        print("Iterative")
        print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
        print("Base")
        print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
        print("Improvement")
        print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))

    random.seed(42)
    np.random.seed(42)

    # Sonar data set
    current_path = pathlib.Path(__file__).parent.resolve()
    sonar_path = pathlib.Path.joinpath(current_path, 'sonar.csv')
    dataset = read_csv(sonar_path, header=0)
    data = dataset.values
    # separate into input and output columns
    X, y = data[:, :-1], data[:, -1]
    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))

    X = X[:, [7, 40]]
    y = 2 * y - 1

    two_features_path = pathlib.Path.joinpath(data_path, '7_40')
    show_data(X, y, two_features_path)

    print("Features 7 40 \n")
    print(X)

    perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y, optimizer=optimizer)

    print("Iterative")
    print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
    print("Base")
    print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
    print("Improvement")
    print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))

    random.seed(42)
    np.random.seed(42)

    # Sonar data set
    current_path = pathlib.Path(__file__).parent.resolve()
    sonar_path = pathlib.Path.joinpath(current_path, 'sonar.csv')
    dataset = read_csv(sonar_path, header=0)
    data = dataset.values
    # separate into input and output columns
    X, y = data[:, :-1], data[:, -1]
    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))

    X = X[:, [1, 47]]
    y = 2 * y - 1

    two_features_path = pathlib.Path.joinpath(data_path, '1_47')
    show_data(X, y, two_features_path)

    print("\n\n\n\n")
    print("Feature 1 47 \n")
    print(X)

    perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y, optimizer=optimizer)

    print("Iterative")
    print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
    print("Base")
    print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
    print("Improvement")
    print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))

    random.seed(42)
    np.random.seed(42)

    # Sonar data set
    current_path = pathlib.Path(__file__).parent.resolve()
    sonar_path = pathlib.Path.joinpath(current_path, 'sonar.csv')
    dataset = read_csv(sonar_path, header=0)
    data = dataset.values
    # separate into input and output columns
    X, y = data[:, :-1], data[:, -1]
    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))

    X = X[:, [10, 44]]
    y = 2 * y - 1

    two_features_path = pathlib.Path.joinpath(data_path, '10_44')
    show_data(X, y, two_features_path)

    print("\n\n\n\n")
    print("Feaure 10 44\n")
    print(X)

    perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y, optimizer=optimizer)

    print("Iterative")
    print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
    print("Base")
    print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
    print("Improvement")
    print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))

    random.seed(42)
    np.random.seed(42)

    # Sonar data set
    current_path = pathlib.Path(__file__).parent.resolve()
    sonar_path = pathlib.Path.joinpath(current_path, 'sonar.csv')
    dataset = read_csv(sonar_path, header=0)
    data = dataset.values
    # separate into input and output columns
    X, y = data[:, :-1], data[:, -1]
    # ensure inputs are floats and output is an integer label
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))

    X = X[:, [11, 35]]
    y = 2 * y - 1

    two_features_path = pathlib.Path.joinpath(data_path, '11_35')
    show_data(X, y, two_features_path)

    print("\n\n\n\n")
    print("Features 11 35 \n")
    print(X)

    perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y, optimizer=optimizer)

    print("Iterative")
    print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
    print("Base")
    print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
    print("Improvement")
    print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))


def run_breast(optimizer, grid):
    random.seed(42)
    np.random.seed(42)

    current_path = pathlib.Path(__file__).parent.resolve()
    data_path = pathlib.Path.joinpath(current_path, 'data_breast')
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype('float32')

    y = 2 * y - 1

    if not grid:
        print("Full \n")
        print(X)

        perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y,
                                                                                                optimizer=optimizer)

        print("Iterative")
        print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
        print("Base")
        print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
        print("Improvement")
        print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))

    random.seed(42)
    np.random.seed(42)

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype('float32')

    X = X[:, [1, 5]]
    y = 2 * y - 1

    two_features_path = pathlib.Path.joinpath(data_path, '1_5')
    show_data(X, y, two_features_path)

    print("Features 1 5 \n")
    print(X)

    perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y, optimizer=optimizer)

    print("Iterative")
    print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
    print("Base")
    print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
    print("Improvement")
    print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))

    random.seed(42)
    np.random.seed(42)

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype('float32')

    X = X[:, [4, 26]]
    y = 2 * y - 1

    two_features_path = pathlib.Path.joinpath(data_path, '4_26')
    show_data(X, y, two_features_path)

    print("\n\n\n\n")
    print("Feature 4 26 \n")
    print(X)

    perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y, optimizer=optimizer)

    print("Iterative")
    print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
    print("Base")
    print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
    print("Improvement")
    print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))

    random.seed(42)
    np.random.seed(42)

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype('float32')

    X = X[:, [3, 23]]
    y = 2 * y - 1

    two_features_path = pathlib.Path.joinpath(data_path, '3_23')
    show_data(X, y, two_features_path)

    print("\n\n\n\n")
    print("Feaure 3 23 \n")
    print(X)

    perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y, optimizer=optimizer)

    print("Iterative")
    print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
    print("Base")
    print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
    print("Improvement")
    print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))

    random.seed(42)
    np.random.seed(42)

    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype('float32')

    X = X[:, [13, 22]]
    y = 2 * y - 1

    two_features_path = pathlib.Path.joinpath(data_path, '13_22')
    show_data(X, y, two_features_path)

    print("\n\n\n\n")
    print("Features 13 22 \n")
    print(X)

    perf_box_cox, standard_dev_box_cox, perf_base, standard_dev_base = evaluate_performance(X, y, optimizer=optimizer)

    print("Iterative")
    print(re.sub('[][]', '', np.array2string(perf_box_cox * 100, precision=3, separator=' & ')))
    print("Base")
    print(re.sub('[][]', '', np.array2string(perf_base * 100, precision=3, separator=' & ')))
    print("Improvement")
    print(re.sub('[][]', '', np.array2string((perf_box_cox - perf_base) * 100, precision=3, separator=' & ')))


if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    # Required positional arguments
    parser.add_argument('dataset_arg', type=str, help='specify dataset')
    parser.add_argument('optimizer', type=int,
                        help='specify if holdout should be use during optimization')

    # Optional arguments
    parser.add_argument('--number_lambdas', type=int, required=False,
                        help='number of lambdas for gridsearch. Default 11')
    parser.add_argument('--epochs', type=int, required=False,
                        help='number of epochs for gridsearch. Default 4')
    parser.add_argument('--shift', type=int, required=False,
                        help='number of shifts for gridsearch. Default = epochs')
    parser.add_argument('--shuffle', type=int, required=False,
                        help='number of shuffles for gridsearch. Default = epochs')
    parser.add_argument('--finer', type=int, required=False,
                        help='number of finer for gridsearch. Default = epochs')

    args = parser.parse_args()

    number_lambdas = 11
    if args.number_lambdas is not None:
        number_lambdas = args.number_lambdas

    epochs = 4
    if args.epochs is not None:
        epochs = args.epochs

    shift = epochs
    if args.shift is not None:
        shift = args.shift

    shuffle = epochs
    if args.shuffle is not None:
        shuffle = args.shuffle

    finer = epochs
    if args.finer is not None:
        finer = args.finer

    print("python version:")
    print(sys.version)
    print("scikit version:")
    print(sklearn.__version__)
    print("numpy version:")
    print(np.__version__)
    print("scipy version:")
    print(scipy.__version__)

    opt = None
    gridsearch = False
    if args.optimizer == 0:
        print("Iterative")
        print("number of lambdas = " + str(number_lambdas))
        print("epochs = " + str(epochs))
        print("shifts = " + str(shift))
        print("shuffle = " + str(shuffle))
        print("finer = " + str(finer))

        opt = IterativeOptimizer(nr_lambdas=number_lambdas,
                                 epochs=epochs,
                                 shift_epoch=shift,
                                 shuffle_epoch=shuffle,
                                 finer_epoch=finer)

    elif args.optimizer == 1:
        print("Gridsearch")
        opt = Gridsearch2D(nr_points=number_lambdas)
        gridsearch = True
    else:
        exit("optimizer does not exist")

    if args.dataset_arg == "sonar":
        run_sonar(opt, gridsearch)
    elif args.dataset_arg == "breast":
        run_breast(opt, gridsearch)
    else:
        exit("dataset does not exist")
