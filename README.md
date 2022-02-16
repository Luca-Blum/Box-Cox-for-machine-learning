# Box-Cox

This repository contains the code that was used for the paper 
[Impact of Box Cox Transformation on Machine Learning Algorithms](Impact_of_Box_Cox_Transformation_on_Machine_Learning_Algorithms_iterative.pdf).

## [Impact of Box-Cox Transformation on 2D Data](boxcox/impact_2D)

Demonstrates the influence of the Box-Cox transformation for artificially generated 
2D dataset. We demonstrated a consistent improvement of the accuracy. Further, we concluded that the accuracy of the classification is dependent on the 
dataset and the classifier itself. Next, it is beneficial to use *full* optimization,
which means that the optimization of the p-dimensional parameter vector for the Box-Cox transformation 
needs to be optimized depend on the full dataset. Hence, one can not optimize each column independently.

To run the experiments first create a folder "logs" and then use:

    python3 -m experiments.2d  > logs/2D.txt

This will create for every dataset a folder. Each folder has a data subfolder that stores different measurements from 
the stratified 10-fold crossvalidation with 5 repetitions. Additionally, a scatter plot of the data is included. Next a 
folder for the heatmaps is created. They were essential to demonstrate the above-mentioned conclusion. Further a folder 
with the influence of the Box-Cox transformation on the data is created. One can see that the transformation skews the 
data in the direction of varying lambda 2. Finally, a performance folder is contained that shows the performance
of all classifiers for varying lambda 2. 


## [Optimization](boxcox/optimization)
This experiment was used to test the iterative optimization for two real world datasets. Additionally, 2-dimensional
subsets were created to compare the method against a gridserch. The scripts need as input the dataset (sonar/breast) and 
the optimization procedure (0=iterative, 1=gridsearch). Additionally, hyperparameters can be added. 
The output first shows the version of the used libraries for reproducibility.
After that the results for the given real world dataset are displayed, followed by the 2-dimensional subset results.
For each of these (sub-) dataset, the dataset itself is printed, then for every classifier, each accuracy measurement of 
the stratified 10-fold crossvalidation with 5 repetitions evaluated on the testset, 
followed by the mean of these measurements and the standard deviation in brackets. Additionally, the mean of the
base classifier and the corresponding standard deviation are given. At the end of a (sub-) dataset everything is summarized
 in a row for the accuracy of all classifiers for the iterative optimization, the accuracy of the base classifiers and the
 corresponding improvements

To run the experiment first create a folder "logs" and then run:
    
    Sonar iterative:
    python3 -m experiments.case_studies sonar 0 --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs/sonar_iter_4_rounds.txt; 
    python3 -m experiments.case_studies sonar 0 --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs/sonar_iter_8_rounds_4_shifts.txt
    python3 -m experiments.case_studies sonar 0 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs/sonar_iter_8_rounds_2_shuffles.txt
    python3 -m experiments.case_studies sonar 0 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs/sonar_iter_8_rounds_4_finer.txt
    python3 -m experiments.case_studies sonar 0 --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs/sonar_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m experiments.case_studies sonar 0 --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs/sonar_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Breast iterative
    python3 -m experiments.case_studies breast 0 --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs/breast_iter_4_rounds.txt
    python3 -m experiments.case_studies breast 0 --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs/breast_iter_8_rounds_4_shifts.txt
    python3 -m experiments.case_studies breast 0 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs/breast_iter_8_rounds_2_shuffles.txt
    python3 -m experiments.case_studies breast 0 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs/breast_iter_8_rounds_4_finer.txt
    python3 -m experiments.case_studies breast 0 --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs/breast_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m experiments.case_studies breast 0 --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs/breast_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Sonar gridsearch:
    python3 -m experiments.case_studies sonar 1 --number_lambdas 11 > logs/sonar_grid.txt

    Breast gridsearch:
    python3 -m experiments.case_studies breast 1 --number_lambdas 11 > logs/breast_grid.txt
