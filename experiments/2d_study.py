import pathlib
import sys
import numpy as np
import scipy
import sklearn
from sklearn.datasets import make_gaussian_quantiles, make_moons, make_blobs, make_classification
from sklearn.preprocessing import MinMaxScaler
from boxcox.impact_2D import Study2D


def evaluate_repeated(name, repetitions=10):
    np.random.seed(42)

    current_path = pathlib.Path(__file__).parent.resolve()

    folder_path = pathlib.Path.joinpath(current_path, name)
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    nr_lambdas = 11
    repeated_performance = np.zeros((nr_lambdas, nr_lambdas, 5))
    repeated_lambdas = None
    repeated_pipelines = None

    for i in range(repetitions):

        if name == "2D_class":
            features, labels = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                                                   n_repeated=0, n_classes=2, random_state=i)
        elif name == "2D_egg":
            features, labels = make_gaussian_quantiles(cov=3., n_samples=1000, n_features=2, n_classes=2,
                                                       random_state=i)
        elif name == "2D_moons":
            features, labels = make_moons(n_samples=1000, noise=0.1, random_state=i)
        elif name == "2D_blobs":
            features, labels = make_blobs(n_samples=1000, centers=[(1, 1), (2, 2)], random_state=i)
        else:
            exit("Dataset does not exist. Please choose [2D_class, 2D_egg, 2D_moons, 2D_blobs")
        labels = 2 * labels - 1

        init_method = 'box_cox'

        data_path = pathlib.Path.joinpath(folder_path, 'data_' + str(i))
        pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

        perf_plot_path = pathlib.Path.joinpath(folder_path, 'performance_plots_' + str(i))
        pathlib.Path(perf_plot_path).mkdir(parents=True, exist_ok=True)

        data_plot_path = pathlib.Path.joinpath(folder_path, 'influence_boxcox_data_plots_' + str(i))
        pathlib.Path(data_plot_path).mkdir(parents=True, exist_ok=True)

        heatmap_path = pathlib.Path.joinpath(folder_path, 'heatmaps_' + str(i))
        pathlib.Path(heatmap_path).mkdir(parents=True, exist_ok=True)

        histogram_path = pathlib.Path.joinpath(folder_path, 'histograms_' + str(i))
        pathlib.Path(histogram_path).mkdir(parents=True, exist_ok=True)

        study = Study2D(name, lower_bound=-5, upper_bound=5)

        features = MinMaxScaler(feature_range=(1, 2)).fit_transform(features, labels)

        performance, standard_deviations, performance_base, stds_base, lambdas, gauss, skew, gauss_base, skew_base, \
        pipelines, description = study.evaluate(features, labels, init_method, nr_lambdas=nr_lambdas,
                                                data_path=data_path)

        study.plot_performance(performance, performance_base, lambdas, pipelines, perf_plot_path)
        study.plot_heatmap_sns(performance, lambdas, pipelines, heatmap_path)
        study.plot_boxcox_influence_data(features, labels, performance, pipelines, lambdas, data_plot_path)
        study.plot_histograms(features, labels, performance, pipelines, lambdas, histogram_path)
        study.show_data(features, labels, name=name, store_path=data_path)

        repeated_performance += performance
        repeated_lambdas = lambdas
        repeated_pipelines = pipelines

    repeated_performance = 1./repetitions * repeated_performance

    heatmap_path = pathlib.Path.joinpath(folder_path, 'repeated_heatmaps')
    pathlib.Path(heatmap_path).mkdir(parents=True, exist_ok=True)

    study.plot_heatmap_sns(repeated_performance, repeated_lambdas, repeated_pipelines, heatmap_path)


def evaluate(features, labels, name):
    np.random.seed(42)

    labels = 2 * labels - 1

    init_method = 'box_cox'

    current_path = pathlib.Path(__file__).parent.resolve()

    folder_path = pathlib.Path.joinpath(current_path, name)
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    data_path = pathlib.Path.joinpath(folder_path, 'data')
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)

    perf_plot_path = pathlib.Path.joinpath(folder_path, 'performance_plots')
    pathlib.Path(perf_plot_path).mkdir(parents=True, exist_ok=True)

    data_plot_path = pathlib.Path.joinpath(folder_path, 'influence_boxcox_data_plots')
    pathlib.Path(data_plot_path).mkdir(parents=True, exist_ok=True)

    heatmap_path = pathlib.Path.joinpath(folder_path, 'heatmaps')
    pathlib.Path(heatmap_path).mkdir(parents=True, exist_ok=True)

    histogram_path = pathlib.Path.joinpath(folder_path, 'histograms')
    pathlib.Path(histogram_path).mkdir(parents=True, exist_ok=True)

    study = Study2D(name, lower_bound=-5, upper_bound=5)

    features = MinMaxScaler(feature_range=(1, 2)).fit_transform(features, labels)

    performance, standard_deviations, performance_base, stds_base, lambdas, gauss, skew, gauss_base, skew_base, \
    pipelines, description = study.evaluate(features, labels, init_method, nr_lambdas=11, data_path=data_path)

    study.plot_performance(performance, performance_base, lambdas, pipelines, perf_plot_path)
    study.plot_heatmap_sns(performance, lambdas, pipelines, heatmap_path)
    study.plot_boxcox_influence_data(features, labels, performance, pipelines, lambdas, data_plot_path)
    study.plot_histograms(features, labels, performance, pipelines, lambdas, histogram_path)
    study.show_data(features, labels, name=name, store_path=data_path)


if __name__ == '__main__':

    print("python version:")
    print(sys.version)
    print("scikit version:")
    print(sklearn.__version__)
    print("numpy version:")
    print(np.__version__)
    print("scipy version:")
    print(scipy.__version__)

    evaluate_repeated("2D_class")

    X1, y1 = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                 n_classes=2, random_state=1)

    evaluate(X1, y1, name="2D_class")

    X2, y2 = make_gaussian_quantiles(cov=3., n_samples=1000, n_features=2, n_classes=2, random_state=1)

    evaluate(X2, y2, name="2D_egg")

    X3, y3 = make_moons(n_samples=1000, noise=0.1, random_state=1)

    evaluate(X3, y3, name="2D_moons")

    X4, y4 = make_blobs(n_samples=1000, centers=[(1, 1), (2, 2)], random_state=1)

    evaluate(X4, y4, name="2D_blobs")

    # How to load saved data
    data_name = "2D_class"

    study_load = Study2D(name=data_name, lower_bound=-5, upper_bound=5)

    path_current = pathlib.Path(__file__).parent.resolve()
    path_data = pathlib.Path.joinpath(path_current, data_name, 'data')

    performance, standard_deviations, performance_base, stds_base, lambdas, gauss, skew, gauss_base, skew_base, \
      pipelines, description = study_load.load_data(path_data)
