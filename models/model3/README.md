#Model 3

This is my LeNet-5 ([http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)) implementation in tensorflow. This is the model I spent most time on, and did a lot of data preprocessing for it. First of all, I do the rotations I described in the past models. Also, if a class has less than 35 members, I perform some random transformations in the original images so that the image looks a little bit different but not different enough that the class is irrecognizable. The preprocessing might be a little slow, and thats why I print some messages to alert about the state of the preprocessing.

