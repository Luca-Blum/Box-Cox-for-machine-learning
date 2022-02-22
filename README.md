# Box-Cox

This repository contains the code that was used for the paper 
[Impact of Box Cox Transformation on Machine Learning Algorithms](Impact_of_Box_Cox_Transformation_on_Machine_Learning_Algorithms_iterative.pdf). A study conducted at ETH Zürich and conceived by:

- Luca Blum <luca.bluem95@gmail.com>
- Dr. Mohamed Elgendi <moe.elgendi@gmail.com>
- Prof. Dr. Carlo Menon <carlo.menon@hest.ethz.ch>



## [Box-Cox](boxcox)
This folder contains the code to study the impact of the Box-Cox transformation for 2D data and the iterative optimization.

## [Experiments](experiments)

This folder contains the scripts for the experiments.

### [Impact of Box-Cox Transformation on 2D Data](boxcox/impact_2D)

Demonstrates the influence of the Box-Cox transformation for artificially generated 
2D dataset. We demonstrated a consistent improvement of the accuracy. Further, we concluded that the accuracy of the classification is dependent on the 
dataset and the classifier itself. Next, it is beneficial to use *full* optimization,
which means that the optimization of the p-dimensional parameter vector for the Box-Cox transformation 
needs to be optimized depend on the full dataset. Hence, one can not optimize each column independently.

To run the experiments first create a folder "logs" and then use:

    python3 -m experiments.2d_study  > logs/2D.txt

This will create for every dataset a folder. Each folder has a data subfolder that stores different measurements from 
the stratified 10-fold crossvalidation with 5 repetitions. Additionally, a scatter plot of the data is included. Next a 
folder for the heatmaps is created. They were essential to demonstrate the above-mentioned conclusion. Further a folder 
with the influence of the Box-Cox transformation on the data is created. One can see that the transformation skews the 
data in the direction of varying lambda 2. Finally, a performance folder is contained that shows the performance
of all classifiers for varying lambda 2. 


### [Optimization](boxcox/optimization)
This experiment was used to test the iterative optimization for two real world datasets. Additionally, 2-dimensional
subsets were created to compare the method against a 2D grid search. The scripts need as input the name of the dataset 
(sonar/breast), the optimization procedure (0=iterative, 1=grid search, 2=mle) and a metric to measure the performance. 
Optionally, hyperparameters can be added. The output first shows the version of the used libraries for reproducibility 
(Python, Scikit, Numpy, Scipy). After that the results for the given real world dataset are displayed, followed by the 
2-dimensional subset results. For each of these (sub-) dataset, the dataset itself is printed, 
then for every classifier, the performance is measured with the specified metric using a 
stratified 10-fold cross-validation with 5 repetitions  evaluated on the test set, 
followed by the mean of these measurements and the standard deviation in brackets. 
Additionally, the mean of the base classifier and the corresponding standard deviation are given. 
At the end of a (sub-) dataset everything is summarized in a row for the performance of all classifiers 
for the specified optimization, the accuracy of the base classifiers and the corresponding improvements.

To run all experiments first create folders "logs_acc". "logs_f1" and "logs_matthews" and then run:
    
    EVALUATE ACCURACY

    Sonar iterative:
    python3 -m experiments.case_studies sonar 0 --metric accuracy --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs_acc/sonar_iter_4_rounds.txt
    python3 -m experiments.case_studies sonar 0 --metric accuracy --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs_acc/sonar_iter_8_rounds_4_shifts.txt
    python3 -m experiments.case_studies sonar 0 --metric accuracy --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs_acc/sonar_iter_8_rounds_2_shuffles.txt
    python3 -m experiments.case_studies sonar 0 --metric accuracy --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs_acc/sonar_iter_8_rounds_4_finer.txt
    python3 -m experiments.case_studies sonar 0 --metric accuracy --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs_acc/sonar_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m experiments.case_studies sonar 0 --metric accuracy --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs_acc/sonar_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Breast iterative
    python3 -m experiments.case_studies breast 0 --metric accuracy --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs_acc/breast_iter_4_rounds.txt
    python3 -m experiments.case_studies breast 0 --metric accuracy --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs_acc/breast_iter_8_rounds_4_shifts.txt
    python3 -m experiments.case_studies breast 0 --metric accuracy --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs_acc/breast_iter_8_rounds_2_shuffles.txt
    python3 -m experiments.case_studies breast 0 --metric accuracy --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs_acc/breast_iter_8_rounds_4_finer.txt
    python3 -m experiments.case_studies breast 0 --metric accuracy --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs_acc/breast_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m experiments.case_studies breast 0 --metric accuracy --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs_acc/breast_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Sonar gridsearch:
    python3 -m experiments.case_studies sonar 1 --metric accuracy --number_lambdas 11 > logs_acc/sonar_grid.txt

    Breast gridsearch:
    python3 -m experiments.case_studies breast 1 --metric accuracy --number_lambdas 11 > logs_acc/breast_grid.txt

    Sonar MLE:
    python3 -m experiments.case_studies sonar 2 --metric accuracy > logs_acc/sonar_mle.txt

    Breast MLE:
    python3 -m experiments.case_studies breast 2 --metric accuracy > logs_acc/breast_mle.txt


    EVALUATE F1 SCORE

    Sonar iterative:
    python3 -m experiments.case_studies sonar 0 --metric f1 --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs_f1/sonar_iter_4_rounds.txt
    python3 -m experiments.case_studies sonar 0 --metric f1 --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs_f1/sonar_iter_8_rounds_4_shifts.txt
    python3 -m experiments.case_studies sonar 0 --metric f1 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs_f1/sonar_iter_8_rounds_2_shuffles.txt
    python3 -m experiments.case_studies sonar 0 --metric f1 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs_f1/sonar_iter_8_rounds_4_finer.txt
    python3 -m experiments.case_studies sonar 0 --metric f1 --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs_f1/sonar_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m experiments.case_studies sonar 0 --metric f1 --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs_f1/sonar_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Breast iterative
    python3 -m experiments.case_studies breast 0 --metric f1 --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs_f1/breast_iter_4_rounds.txt
    python3 -m experiments.case_studies breast 0 --metric f1 --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs_f1/breast_iter_8_rounds_4_shifts.txt
    python3 -m experiments.case_studies breast 0 --metric f1 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs_f1/breast_iter_8_rounds_2_shuffles.txt
    python3 -m experiments.case_studies breast 0 --metric f1 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs_f1/breast_iter_8_rounds_4_finer.txt
    python3 -m experiments.case_studies breast 0 --metric f1 --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs_f1/breast_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m experiments.case_studies breast 0 --metric f1 --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs_f1/breast_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Sonar gridsearch:
    python3 -m experiments.case_studies sonar 1 --metric f1 --number_lambdas 11 > logs_f1/sonar_grid.txt

    Breast gridsearch:
    python3 -m experiments.case_studies breast 1 --metric f1 --number_lambdas 11 > logs_f1/breast_grid.txt

    Sonar MLE:
    python3 -m experiments.case_studies sonar 2 --metric f1 > logs_f1/sonar_mle.txt

    Breast MLE:
    python3 -m experiments.case_studies breast 2 --metric f1 > logs_f1/breast_mle.txt


    EVALUATE MATTHEWS CROSS CORRELATION

    Sonar iterative:
    python3 -m experiments.case_studies sonar 0 --metric matthews --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs_matthews/sonar_iter_4_rounds.txt
    python3 -m experiments.case_studies sonar 0 --metric matthews --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs_matthews/sonar_iter_8_rounds_4_shifts.txt
    python3 -m experiments.case_studies sonar 0 --metric matthews --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs_matthews/sonar_iter_8_rounds_2_shuffles.txt
    python3 -m experiments.case_studies sonar 0 --metric matthews --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs_matthews/sonar_iter_8_rounds_4_finer.txt
    python3 -m experiments.case_studies sonar 0 --metric matthews --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs_matthews/sonar_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m experiments.case_studies sonar 0 --metric matthews --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs_matthews/sonar_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Breast iterative
    python3 -m experiments.case_studies breast 0 --metric matthews --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs_matthews/breast_iter_4_rounds.txt
    python3 -m experiments.case_studies breast 0 --metric matthews --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs_matthews/breast_iter_8_rounds_4_shifts.txt
    python3 -m experiments.case_studies breast 0 --metric matthews --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs_matthews/breast_iter_8_rounds_2_shuffles.txt
    python3 -m experiments.case_studies breast 0 --metric matthews --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs_matthews/breast_iter_8_rounds_4_finer.txt
    python3 -m experiments.case_studies breast 0 --metric matthews --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs_matthews/breast_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m experiments.case_studies breast 0 --metric matthews --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs_matthews/breast_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Sonar gridsearch:
    python3 -m experiments.case_studies sonar 1 --metric matthews --number_lambdas 11 > logs_matthews/sonar_grid.txt

    Breast gridsearch:
    python3 -m experiments.case_studies breast 1 --metric matthews --number_lambdas 11 > logs_matthews/breast_grid.txt

    Sonar MLE:
    python3 -m experiments.case_studies sonar 2 --metric matthews > logs_matthews/sonar_mle.txt

    Breast MLE:
    python3 -m experiments.case_studies breast 2 --metric matthews > logs_matthews/breast_mle.txt






