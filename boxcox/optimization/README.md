# Optimization

To optimize the Boxcox transformation for classification [classifierOptimization](classifierOptimization.py)
 was developed as an interface to scikit-learn. Hence, an object of this class can be directly used like an usual scikit-learn
 classifier. Therefore, basic pipelines can be constructed.

[lambda_optimizer](lamba_optimizer.py) is an abstract class that is used to get the optimized lambda parameters 
for the Box-Cox transformation during the training phase of the classifier. 

[iterative_optimization](iterative_optimization.py) is a concrete class that implements the [lambda_optimizer](lamba_optimizer.py).
 An iterative optimization is used to get optimal lambda parameters.


[gridsearch](gridsearch2D.py) is a concrete class that implements the [lambda_optimizer](lamba_optimizer.py). 
 It uses a gridsearch and can only be used for 2D dataset, because the computational costs increases quickly for higher dimensions.

Further optimization methods can be directly added by implementing the [lambda_optimizer](lamba_optimizer.py) abstract class.