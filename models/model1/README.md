#Model 1

This one is the logistic regressor implemented in scikit-learn. This model uses the solver saga, and it trains in 10000 iterations with the amount of processors just computer has (automatically takes all available processors using n_jobs=-1.

In the preprocessing of the data, we extract add to the training set some rotations of certain data and we extract 10 Shi-Thomasi corners as the features of each image.