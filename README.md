# German-Traffic-Signs-Detector
Kiwi Campus Deep Learning Challange. Solution by Alejandro Calle Saldarriaga.

This code is a click application with several functionalities:

**download**

First of all it downloads the *German Traffic Signs Dataset* ([http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset)) and stores the data set in the *images* folder accordingly, with 80% of the data in *images/train/* and the remaining images in *image/test/*.

To use it you just need to call:

    python app.py download

**train**

Then you have the train funcionality. It takes a model name and a folder where train images are located and trains the model you choose, saving it in the corresponding directory. There are three models to choose from:

'linear-scikit': This is a scikit implementation of a logistic regression model. A typical call would be:

    python app.py train -m 'linear-scikit' -d './images/train'

'linear-tensorflow': This is the TensorFlow implementation of a logistic regression model. Also uses the test data in './images/test' for testing, read the model's readme to know. A typical call would be:

    python app.py train -m 'linear-tensorflow' -d './images/train'

'lenet': This is the TensorFlow implementation of LeNet-5 (described in ([http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf))). A typical call would be:

    python app.py test -m 'lenet' -d './images/train'

**test**

You also have the test functionality, that will take one of the models trained in the previous step and test their accuracy using test images. We have two models to choose from:

'linear-scikit': Will load the scikit logistic regression and compute the accuracy. A typical call is:

    python app.py test -m 'linear-scikit' -d './images/test'

'lenet': Will load LeNet and will predict on test data and see how accurate it is. A typical call is:

    python app.py test -m 'lenet' -d './images/test'

**infer**

The last functionality is the infer function. It takes one of the models trained in the train step and infers the classes of some new images. It shows the image + the inferred label on screen. It takes a few seconds to change image, or you just can press a key on your keyboard to change the image. Only implemented for 'linear-scikit' and 'lenet'.

Typical uses:

    python app.py infer -m 'linear-scikit' -d './images/user'

    python app.py infer -m 'lenet' -d './images/user'



