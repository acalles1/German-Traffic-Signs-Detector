#Model 2

This one is the logistic regressor implemented in tensorflow. This model uses the tensorflow estimator API. It creates some .csv momentary files for data parsing/feeding to the model but then erases them. It feed the model the same features of the last model.

It trains with a batch size of 20 for 1000 epochs.

Remember, this model does not have a test or infer option. The test data is deployed when you call the train function. This is because I was not able to figure out how to load a model from the estimator API (weirdly enough I was able to save it, but loading it is another beast altogether).