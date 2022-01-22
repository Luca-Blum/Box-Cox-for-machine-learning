# Box-Cox

This repository contains the code that was used for the paper 
[Impact of Box Cox Transformation on Machine Learning Algorithms](Impact_of_Box_Cox_Transformation_on_Machine_Learning_Algorithms_iterative.pdf).

## [Optimization](boxcox/optimization)

To run the experiments first create a folder "logs" and then run:
    
    Sonar iterative:
    python3 -m optimization.optimization.experiments sonar 0 --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs/sonar_iter_4_rounds.txt; 
    python3 -m optimization.optimization.experiments sonar 0 --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs/sonar_iter_8_rounds_4_shifts.txt
    python3 -m optimization.optimization.experiments sonar 0 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs/sonar_iter_8_rounds_2_shuffles.txt
    python3 -m optimization.optimization.experiments sonar 0 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs/sonar_iter_8_rounds_4_finer.txt
    python3 -m optimization.optimization.experiments sonar 0 --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs/sonar_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m optimization.optimization.experiments sonar 0 --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs/sonar_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Breast iterative
    python3 -m optimization.optimization.experiments breast 0 --number_lambdas 11 --epochs 4 --shift 4 --shuffle 4 --finer 4 > logs/breast_iter_4_rounds.txt
    python3 -m optimization.optimization.experiments breast 0 --number_lambdas 11 --epochs 8 --shift 4 --shuffle 8 --finer 8 > logs/breast_iter_8_rounds_4_shifts.txt
    python3 -m optimization.optimization.experiments breast 0 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 2 --finer 8 > logs/breast_iter_8_rounds_2_shuffles.txt
    python3 -m optimization.optimization.experiments breast 0 --number_lambdas 11 --epochs 8 --shift 8 --shuffle 8 --finer 4 > logs/breast_iter_8_rounds_4_finer.txt
    python3 -m optimization.optimization.experiments breast 0 --number_lambdas 11 --epochs 16 --shift 8 --shuffle 2 --finer 4 > logs/breast_iter_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m optimization.optimization.experiments breast 0 --number_lambdas 21 --epochs 16 --shift 8 --shuffle 4 --finer 4 > logs/breast_iter_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Sonar gridsearch:
    python3 -m optimization.optimization.experiments sonar 1 --number_lambdas 11 > logs/sonar_grid_4_rounds.txt
    python3 -m optimization.optimization.experiments sonar 1 --number_lambdas 11 > logs/sonar_grid_8_rounds_4_shifts.txt
    python3 -m optimization.optimization.experiments sonar 1 --number_lambdas 11 > logs/sonar_grid_8_rounds_2_shuffles.txt
    python3 -m optimization.optimization.experiments sonar 1 --number_lambdas 11 > logs/sonar_grid_8_rounds_4_finer.txt
    python3 -m optimization.optimization.experiments sonar 1 --number_lambdas 11 > logs/sonar_grid_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m optimization.optimization.experiments sonar 1 --number_lambdas 21 > logs/sonar_grid_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt

    Breast gridsearch:
    python3 -m optimization.optimization.experiments breast 1 --number_lambdas 11 > logs/breast_grid_4_rounds.txt
    python3 -m optimization.optimization.experiments breast 1 --number_lambdas 11 > logs/breast_grid_8_rounds_4_shifts.txt
    python3 -m optimization.optimization.experiments breast 1 --number_lambdas 11 > logs/breast_grid_8_rounds_2_shuffles.txt
    python3 -m optimization.optimization.experiments breast 1 --number_lambdas 11 > logs/breast_grid_8_rounds_4_finer.txt
    python3 -m optimization.optimization.experiments breast 1 --number_lambdas 11 > logs/breast_grid_16_rounds_8_shifts_2_shuffles_4_finer.txt
    python3 -m optimization.optimization.experiments breast 1 --number_lambdas 21 > logs/breast_grid_21_lambdas_16_rounds_8_shifts_2_shuffles_4_finer.txt     

## [impact_2D](boxcox/impact_2D)

Demonstrates the influence of the Box-Cox transformation for artificially generated 
2D dataset. We demonstrated a consistent improvement of the accuracy. Further, we concluded that the accuracy of the classification is dependent on the 
dataset and the classifier itself. Next, it is beneficial to use *full* optimization,
which means that the optimization of the p-dimensional parameter vector for the Box-Cox transformation 
needs to be optimized depend on the full dataset. Hence, one can not optimize each column independenlty.

To run the experiments first create a folder "logs" and then use:

    python3 -m experiments.2d  > logs/2D.txt