'''
Title           :app.py
Description     :Click app that solves Kiwi's deep-learning challenge.
Author          :Alejandro Calle-Saldarriaga.
Date Created    :11/05/18
Date Modified   :
version         :0.1
usage           :
input           :
output          :
python_version  :2.7.13
'''

import click
import os
import requests
import cv2
import random
import csv
import pickle
import tensorflow as tf
import numpy as np
from skimage.transform import rotate, warp, ProjectiveTransform
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework import ops
from tensorflow.contrib.layers import flatten

"""
Utilities:
the next lines are some utility functions I use througout the code, and a list
with the name of the classes for showing images with pretty tags (i.e. the actual
names of the class instead of some number).
"""

classes = ['Speed limit (20 km/h)',
           'Speed limit (30 km/h)',
           'Speed limit (50 km/h)',
           'Speed limit (60 km/h)',
           'Speed limit (70 km/h)',
           'Speed limit (80 km/h)',
           'End of speed limit (80 km/h)',
           'Speed limit (100 km/h)',
           'Speed limit (120 km/h)',
           'No passing',
           'No passing for vehicles over 3.5 metric tons',
           'Right-of-way at the next intersection',
           'Priority road',
           'Yield',
           'Stop',
           'No vehicles',
           'Vehicles over 3.5 metric tons prohibited',
           'No entry',
           'General caution',
           'Dangerous curve to the left',
           'Dangerous curve to the right',
           'Double curve',
           'Bumpy road',
           'Slippery road',
           'Road narrows on the right',
           'Road work',
           'Traffic signals',
           'Pedestrians',
           'Children crossing',
           'Bicycles crossing',
           'Bewareof ice/snow',
           'Wild animals crossing',
           'End of all speed and passing limits',
           'Turn right ahead',
           'Turn left ahead',
           'Ahead only',
           'Go straight or right',
           'Go straight or left',
           'Keep right',
           'Keep left',
           'Roundabout mandatory',
           'End of no passing',
           'End of no passing by vechiles over 3.5 metric tons']

def tanh_lecun(x):
    """
    The special hyperbolic tangent function proposed by LeCun.
    Built with Tensorflow primitives so that tensorflow can use it easily and propagate
    and compute the gradients appropiately.
    The function is f(a) = A*tanh(S*a), with S and A specified below.
    """
    A = tf.constant(1.17159)
    S = tf.constant(2.0/3.0)
    mult = tf.multiply(S, x)
    app = tf.nn.tanh(mult)
    return tf.multiply(A, app)

def lenet(x):
    #Parameters for randomly initliasing the different weight values.
    mu = 0
    sigma = 0.1
    #I'm gonna initialize all weights with mean 0 and 0.1 standard deviation.
    #The weight initliazition procedure proposed in LeCun gives extremely high values and
    #the network becomes inefficient. Maybe it is a good intilization for the digit recognition
    #problem but not for this problem?
    #This is the first convolutional layer C1
    #Initialize weights for the first convolutional layer. 6 feature maps connected to
    #one (1) 5x5 neighborhood in the input. 5*5*1*6=150 trainable parameters
    C1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6], mean=mu, stddev=sigma))
    #Bias for each feature map. 6 parameters, with the weights we have 156 parameters
    C1_b = tf.Variable(tf.zeros(6))
    #Define the convolution layer with the weights and biases defined.
    C1 = tf.nn.conv2d(x, C1_w, strides = [1,1,1,1], padding = 'VALID') + C1_b
    #LeCun uses a sigmoidal activation function here.
    C1 = tanh_lecun(C1)

    #This is the sub-sampling layer S2
    #Subsampling (also known as average pooling) with 2x2 receptive fields. 12 parameters.
    S2 = tf.nn.avg_pool(C1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    #The result is passed to a sigmoidal function
    S2 = tanh_lecun(S2)

    #Another convolutional layer C3.
    #Initlialize weights. 16 feature maps connected connected to 5*5 neighborhoods
    #5*5*6*16=2400+16=2416 trainable parameters. Little difference with LeCun here as they
    #have less parameters to train in this part.
    C3_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean=mu, stddev=sigma))
    #not all feature maps are used, need to split
    C3_b = tf.Variable(tf.zeros(16))
    C3 = tf.nn.conv2d(S2, C3_w, strides = [1,1,1,1], padding = 'VALID') + C3_b
    C3 = tanh_lecun(C3)

    #Sub-sampling layer S2
    S4 = tf.nn.avg_pool(C3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    #Activation
    S4 = tanh_lecun(S4)

    #C5: Flattened with 120 feature maps. Full connection. Labeled as convolutional
    #insted of fully-connected because if LeNet-5 input were made bigger with
    #everything else kept constat the feature map dimension would be larger than 1x1.
    #Shape = 400 since each unit is connected to 5x5 neighbors on the 16 features maps
    #of S4.
    C5 = flatten(S4)
    C5_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean=mu, stddev=sigma))
    C5_b = tf.Variable(tf.zeros(120))
    C5 = tf.matmul(C5,C5_w) + C5_b
    #Activation
    C5 = tanh_lecun(C5)

    #Fully connected with 84 units. Has 10164 trainable parameters 120*84 + 84 = 10164
    F6_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean=mu, stddev=sigma))
    F6_b = tf.Variable(tf.zeros(84))
    F6 = tf.matmul(C5,F6_w) + F6_b
    # Activation
    F6 = tanh_lecun(F6)

    # Output Layer: shape 84x43 (84 from the last layer, 43 because that is the number of
    # classes in our classification problem).
    out_w = tf.Variable(tf.truncated_normal(shape = (84,43), mean=mu, stddev=sigma))
    out_b = tf.Variable(tf.zeros(43))
    out = tf.matmul(F6, out_w) + out_b
    return out

def export_model(model, columns, export_dir):
    """
    Export to SavedModel format. For TensorFlow estimators. This was a function I created
    in order to save the Tensorflow linear estimator. I was able to save it... but wasn't able
    to load it correctly for testing/inferring later. I'm gonna leave it here in case I figure it
    out later.
    """
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    model.export_savedmodel(export_dir, example_input_fn)

def my_input_fn(file_path, num_epochs, perform_shuffle, batch_size, default, feature_names):
    """
    This function is a helper I build in order to be able to define the linear classifier
    in tensorflow. Reads an csv file and creates the appropiate tensors in batches
    """
    def decode_csv(line):
       """
       Parses the csv and creates the appropiate tensor structure for the labels and
       a dictionary for features and their values
       """
       parsed_line = tf.decode_csv(line, default)
       label = parsed_line[-1:] # Last element is the label
       del parsed_line[-1] # Delete last element
       features = parsed_line # Everything (but last element) are the features
       d = dict(zip(feature_names, features)), label
       return d

    dataset = (tf.data.TextLineDataset(file_path) # Read text file
        .skip(1) # Skip header row
        .map(decode_csv)) #decode each line and converts it appropiately
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(num_epochs) # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def rotate_img(x, y, path, cla, lenet=False):
    """
    Rotates all images in a given class, appends it to appropiate structure
    """
    #All of LeNet's images are scaled like this
    scaler = MinMaxScaler(feature_range=(-0.1, 1.175))
    all_img = os.listdir(path)
    #I need a list structure in order to .append
    aux = x.tolist()
    for img in all_img:
        if int(img[0:2]) == cla:
            image_path = path + '/' + img
            image_read = cv2.imread(image_path, 0) #read in greyscale
            rows , cols = image_read.shape
            #Estimate the rotation matrix
            M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
            #Actually rotate the image
            dst = cv2.warpAffine(image_read, M , (cols,rows))
            #Equalize histograms, this gives a 'clearer' image that will be better input for LeNet
            equalized = cv2.equalizeHist(dst)
            if lenet:
                # I use this method both for creating LeNet's database and for the other two
                # databases. Since LeNet recieves the images themselves in 32x32 format and
                # the other two recieves some features, I need two different approaches.
                resize = cv2.resize(image_read, (32, 32), interpolation=cv2.INTER_CUBIC)
                X_new = scaler.fit_transform(resize)
                y.append(int(cla))
                X = np.array(X_new)
                X = np.reshape(X, (32, 32, 1))
                #look that X is an image itself
                aux.append(X)
            else:
                corners = cv2.goodFeaturesToTrack(equalized, 10, 1e-80, 1)
                #flatten list to correctly pass it to x_train
                flat_list = [item for sublist in corners for item in sublist]
                #Need to this two steps to flatten again correctly because of the way
                #opencv saves points they extract.
                test = np.array(flat_list)
                flatter = [item for subarray in test for item in subarray]
                #flatter is not an image, it is a set of features
                aux.append(flatter)
                y.append(cla)
        else:
            continue
    return np.array(aux), y

def flip_img(x, y, path, cla, lenet=False):
    """
    Flips all images in a given class, appends it to appropiate structure
    """
    #All of LeNet's images are scaled like this
    scaler = MinMaxScaler(feature_range=(-0.1, 1.175))
    all_img = os.listdir(path)
    #I need a list structure in order to .append
    aux = x.tolist()
    for img in all_img:
        if int(img[0:2]) == cla:
            image_path = path + '/' + img
            image_read = cv2.imread(image_path, 0) #read in greyscale
            flipped = cv2.flip(image_read, 1)
            equalized = cv2.equalizeHist(flipped)
            if lenet:
                # I use this method both for creating LeNet's database and for the other two
                # databases. Since LeNet recieves the images themselves in 32x32 format and
                # the other two recieves some features, I need two different approaches.
                resize = cv2.resize(equalized, (32, 32), interpolation=cv2.INTER_CUBIC)
                X_new = scaler.fit_transform(resize)
                y.append(int(cla))
                X = np.array(X_new)
                X = np.reshape(X, (32, 32, 1))
                #look that X is an image itself
                aux.append(X)
            else:
                corners = cv2.goodFeaturesToTrack(equalized, 10, 1e-80, 1)
                #flatten list to correctly pass it to x_train
                flat_list = [item for sublist in corners for item in sublist]
                #Need to this two steps to flatten again correctly because of the way
                #opencv saves points they extract.
                test = np.array(flat_list)
                flatter = [item for subarray in test for item in subarray]
                #flatter is not an image, it is a set of features
                aux.append(flatter)
                y.append(cla)
        else:
            continue
    return np.array(aux), y

def linear_scikit_train(path):
    """
    Reciebes a path of images (e.g. './images/train') and create the appropiate structure
    for the linear models (x_train and y_train).
    """
    x_train, y_train = create_npy_lin(path)
    n_classes = 43
    min_ex = 50
    #list of classes that are symmetric horizontally or vertically,
    #meaning that if you flip them 180 degrees then they will be the same traffic sign.
    #This were selected manually.
    symmetric_h = [12, 15, 17, 26, 40]
    #All these classes will have double the images then.
    for cla in symmetric_h:
        if y_train[cla] <= min_ex:
            x_train, y_train = rotate_img(x_train, y_train, path, cla)

    #Now this is a list of classes whose convert into another class when flipped
    #they convert to is always the next one.
    #For example: turn left converts in turn right when flipped
    flip_change = [19, 33, 36, 38]
    for cla in flip_change:
        if y_train[cla] <= min_ex:
            #if in the current there are not enough images, rotate from the next
            x_train, y_train = flip_img(x_train, y_train, path, cla+1)
        if y_train[cla+1] <= min_ex:
            #if on the next there are not enough images, rotate from the current
            x_train, y_train = flip_img(x_train, y_train, path, cla)

    #Now images when flipped stay the same class.
    flippable = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35, 40]
    for cla in flippable:
        if y_train[cla] <= min_ex:
            #if in the current there are not enough images, rotate from the next
            x_train, y_train = flip_img(x_train, y_train, path, cla)

    return x_train, y_train

def create_input(path):
    """
    Create Lenet's input with appropiate structure given a path (e.g. './images/train' or
    './images/test')
    """
    folder =  path
    files = os.listdir(folder)
    x = []
    y = []
    image_paths = []
    scaler = MinMaxScaler(feature_range=(-0.1, 1.175))
    #noramlized as in LeCun, makes the mean input roughly 0 and the variance roughly 1.
    #This accelerates learning.
    for i, images in sorted(enumerate(files)):
        label = images[0:2] #class identifier is in these positions
        image_path = folder + '/' + images
        image_paths.append(image_path)
        image_read = cv2.imread(image_path, 0)
        resize = cv2.resize(image_read, (32, 32), interpolation=cv2.INTER_CUBIC)
        X_new = scaler.fit_transform(resize)
        x.append(X_new)
        y.append(int(label))
    X = np.array(x)
    n, m, p = X.shape
    x_aux = []
    for example in X:
        for row in example:
            for element in row:
                x_aux.append([element])
    x_aux = np.array(x_aux)
    x_aux = np.reshape(x_aux, (n, 32, 32, 1))
    return x_aux, y, image_paths

def augment_input(x, y):
    """
    This augments the training input for Lenet. Does rotation on symmetric images, rotation
    for images that when rotated become other class, and some random transofrmations on the
    images. Does this a bunch of times until each class has at least 50 training examples.
    """
    path = './images/train'
    n_classes = 43
    min_ex = 35
    #list of classes that are symmetric horizontally or vertically,
    #meaning that if you flip them 180 degrees then they will be the same traffic sign.
    #This were selected manually.
    symmetric_h = [12, 15, 17, 26, 40]
    #All these classes will have double the images then.
    for cla in symmetric_h:
        if y[cla] <= min_ex:
            x , y = rotate_img(x, y, path, cla, lenet=True)

    #Now this is a list of classes whose convert into another class when flipped
    #they convert to is always the next one.
    #For example: turn left converts in turn right when flipped
    flip_change = [19, 33, 36, 38]
    for cla in flip_change:
        if y[cla] <= min_ex:
            #if in the current there are not enough images, rotate from the next
            x , y = flip_img(x, y, path, cla+1, lenet=True)
        if y[cla+1] <= min_ex:
            #if on the next there are not enough images, rotate from the current
            x, y = flip_img(x, y, path, cla, lenet=True)

    #Now images when flipped stay the same class.
    flippable = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35, 40]
    for cla in flippable:
        if y[cla] <= min_ex:
            #if in the current there are not enough images, rotate from the next
            x, y = flip_img(x, y, path, cla,lenet=True)

    #Note: I print a lot of stuff here because this part is kinda slow, so I print in order to see
    #if everything is running smoothly. It transforms in batches, that is, it transforms all the
    #images in a folder each iteration.
    for cla in range(43):
        #Do random transforms until I have all classes with at least 35 images.
        print('Current class:')
        print(cla)
        counts, unique = np.unique(y, return_counts=True)
        examples = unique[cla]
        i=0
        while examples <= min_ex:
            print('Current amount of batch transformations:')
            print(i)
            x, y = transform_img(x, y, path, cla)
            counts, unique = np.unique(y, return_counts=True)
            examples = unique[cla]
            print('Current amount of images in class:')
            print(examples)
            i+=1

    return x, y

def transform_img(x, y, path, cla):
    """
    Projective transform of all images in a class
    """
    #Scale as in LeCun
    scaler = MinMaxScaler(feature_range=(-0.1, 1.175))
    all_img = os.listdir(path)
    #List structure so I can .append
    aux = x.tolist()
    for img in all_img:
        if int(img[0:2]) == cla:
            image_path = path + '/' + img
            #prepare parameters for randomization
            intensity = 0.75
            image_read = cv2.imread(image_path, 0) #read in greyscale
            resize = cv2.resize(image_read, (32, 32), interpolation=cv2.INTER_CUBIC)
            image_shape = resize.shape
            image_size = image_shape[0]
            d = image_size * 0.3 * intensity
            #With these 8 parameters we can perform a transofrmation of the image in such a way
            #that the image is different enough from the original but not too different, since
            #we should be able to still recognize the class in the transformed image.
            tl_top = random.uniform(-d, d)     # Top left corner, top margin
            tl_left = random.uniform(-d, d)    # Top left corner, left margin
            bl_bottom = random.uniform(-d, d)  # Bottom left corner, bottom margin
            bl_left = random.uniform(-d, d)    # Bottom left corner, left margin
            tr_top = random.uniform(-d, d)     # Top right corner, top margin
            tr_right = random.uniform(-d, d)   # Top right corner, right margin
            br_bottom = random.uniform(-d, d)  # Bottom right corner, bottom margin
            br_right = random.uniform(-d, d)   # Bottom right corner, right margin
            transform = ProjectiveTransform()
            transform.estimate(np.array((
                       (tl_left, tl_top),
                       (bl_left, image_size - bl_bottom),
                       (image_size - br_right, image_size - br_bottom),
                       (image_size - tr_right, tr_top)
                   )), np.array((
                       (0, 0),
                       (0, image_size),
                       (image_size, image_size),
                       (image_size, 0)
                   )))
            warped = warp(image_read,
                       transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')
            X_new = scaler.fit_transform(warped)
            warped = np.reshape(X_new, (32, 32, 1))
            aux.append(warped)
            y.append(cla)
    return np.array(aux), y

def create_npy_lin(folder):
     """
     This function recieves 'train' or 'test' and creates a npy x file for the
     corresponding directory, for Tensorflow's linear model.
     """
     path = folder
     files = os.listdir(path)
     x = []
     y = []
     all_keypoints = []
     for i, images in sorted(enumerate(files)):
         label = images[0:2] #class identifier is in these positions
         #Sorted because we need the same order as the y files we created earlier.
         image_path = path + '/' + images
         image_read = cv2.imread(image_path, 0) #read in greyscale
         equalized = cv2.equalizeHist(image_read)
         #Need to extract some features, I will extract 10 for every image. Remember
         #that some images are very small so 10 points is okay.
         corners = cv2.goodFeaturesToTrack(equalized, 10, 1e-80, 1)
         #flatten list to correctly pass it to x_train
         flat_list = [item for sublist in corners for item in sublist]
         #Need to this two steps to flatten again correctly because of the way
         #opencv saves points they extract.
         test = np.array(flat_list)
         flatter = [item for subarray in test for item in subarray]
         x.append(flatter)
         y.append(label)
     return x, y

#---------------------------------------------------------------------
# --- END OF UTILITIES, THE ACTUAL CLICK APPLICATION BEGINS HERE ---
#---------------------------------------------------------------------

@click.group()
def cli():
    pass

@click.command()
def download():
    """
    Downloads dataset, unzips it and saves it under the correct folders. Creates
    training/test folders and some numpy files whose rows represent the grayscale 32x32
    pixels for each image with the corresponding labels for train and test.
    Also artificially enlargens dataset in order to represent underepresented classes.
    """
    localFilePath = 'dataset.zip'
    url = 'http://ipv4.download.thinkbroadband.com/5MB.zip'
    print('start downloading!')
    r = requests.get(url, allow_redirects=True)
    open(localFilePath, 'wb').write(r.content)
    print('finish downloading!')
    os.system('unzip ' + localFilePath)
    os.system('rm ' + localFilePath)
    #data preparation: separate the dataset into train and test images
    directory = './FullIJCNN2013'
    dirs = os.listdir(directory)
    all_paths = []
    for curr_dir in dirs:
        if len(curr_dir) == 2: #the folders are exactly two characters long
            path = directory + '/' + curr_dir
            images = os.listdir(path)
            random.shuffle(images)
            for i, imag in enumerate(images):
                #this way I guarantee at least 1 image per class in train/test
                label = path[-2:]
                filename_pre = 'hc.ppm' #hard copy identifier
                #not copy over them later.
                filename = label + filename_pre
                if i == 0:
                    os.system('cp ' + path + '/' + imag + ' ./images/train/' + filename)
                elif i == 1:
                    os.system('cp ' + path + '/' + imag + ' ./images/test/' + filename)
                else:
                    image_path = path + '/' + imag
                    all_paths.append(image_path)

    #Now that I have all the image paths in a list (except the images I hard copied), I need to
    #shuffle them in train/test directories
    random.shuffle(all_paths)
    for i, path in enumerate(all_paths):
        filename_pre = path[-9:] #filename identifier
        label = path[-12:-10] #class identifier is in these positions
        ident = '%04d' % i
        filename = label + ident + filename_pre #Avoid same-named files, four digit identifier
        #one fifth of the images will be in test (i.e. 20%)
        if i % 5 == 0:
            os.system('cp ' + path + ' ./images/test/' + filename)
        #the 4/5ths left
        else:
            os.system('cp ' + path + ' ./images/train/' + filename)

    #Files are ordered in train and test directories, no longer need this.
    os.system('rm -r ./FullIJCNN2013')

@click.command()
@click.option('-m', help='choose which model you want to train')
@click.option('-d', help='choose the directory of your training data')
def train(m, d):

    if m == 'linear-scikit':

        x_train, y_train = linear_scikit_train(d)
        lr = LogisticRegression(solver='saga', max_iter=10000, n_jobs=-1)
        lr.fit(x_train, y_train)

        fileobject = open('./models/model1/saved/model1.p', 'w')
        pickle.dump(lr, fileobject)
        fileobject.close()

    if m == 'linear-tensorflow':

        #This is actually train + test since I wasn't actually able to figure out how to load
        #saved a tf.estimator. I can save it but... No idea yet how to load it.
        x_test, y_test = create_npy_lin('./images/test')
        x_train, y_train = linear_scikit_train(d)
        default = []
        num_features = len(x_train[0])
        feature_names = ['feature' + str(i) for i in range(num_features)]
        #I need to tell the csv parser which are the default values for each feature
        for i in range(num_features):
            #the features are floats, ergo the default value is [0.]
            default.append([0.])
        #labels are ints, then this is their default value.
        default.append([0])
        #I need to convert x_train and y_train in a single csv because of the way tensorflow
        #recieves data.
        headers = ['feature' + str(i) for i in range(num_features)]
        headers.append('label')
        #create auxiliary csv for data preparation (the model can't be fed np.arrays so we
        #have to prepare the data accordingly). This is one of the ways, using an csv parser.
        for i, row in enumerate(x_train):
             row.append(y_train[i])
        x_train.insert(0, headers)
        with open("aux.csv",'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(x_train)
        for i, row in enumerate(x_test):
            row.append(y_test[i])
        x_test.insert(0, headers)
        with open("aux2.csv",'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(x_test)

        #TensorFlow is too verbose when training this model, I don't like it
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        file_path = 'aux.csv'
        test_data = 'aux2.csv'
        model_dir = './models/model2/saved'
        batch_size = 20
        num_epochs = 1000
        feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]
        #Initiliaze classifier.
        classifier = tf.estimator.LinearClassifier(
                     model_dir=model_dir,
                     feature_columns=feature_columns,
                     n_classes = 43)
        #Train classifier
        classifier.train(input_fn=lambda:
                         my_input_fn(file_path, num_epochs,
                                     True, batch_size, default, feature_names))

        #Evaluate classifier (i.e. test)
        evaluate_result = classifier.evaluate(
            input_fn=lambda: my_input_fn(test_data, 1, False, 1, default, feature_names))
        print("Evaluation results")
        for key in evaluate_result:
            print("   {}, was: {}".format(key, evaluate_result[key]))
        export_model(classifier, feature_columns, model_dir)

        #Remove auxiliary files.
        os.system('rm aux.csv')
        os.system('rm aux2.csv')

    if m == 'lenet':

        #Create and augment input
        X_train, y_train, _ = create_input(d)
        X_train, y_train = augment_input(X_train, y_train)
        #By the way, I only transform the original images. Transforming transformations might
        #gives us strange images that could be unrecognizable.

        #Hyperparameters
        #I will train 1000 epochs, 40 images per epoch.
        EPOCHS = 1000
        BATCH_SIZE = 40
        #auxiliaries
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, 43) #43x43 Identity Matrix

        #Learning rate, initialize lenet
        rate = 0.001
        out = lenet(x)

        #We will mimize the mean of the softmax cross entropy
        #between the output and the actual label
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        # Optimizer: Adam is a good and quick optimizer. Tensorflow has no Levenberg-Marquardt
        # as they use in LeCun
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)

        #need to initialize saver here in order to be able to correctly save the model.
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)

            print("Training...")
            print()
            for i in range(EPOCHS):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

                print("EPOCH {} ...".format(i+1))
                print()

            saver.save(sess, './models/model3/saved/lenet')
            print("Model saved")

@click.command()
@click.option('-m', help="choose which model you want to test your data on.")
@click.option('-d', help='choose the directory of your training data')
def test(m, d):
    if m == 'linear-scikit':

         x_test, y_test = create_npy_lin(d)
         with open('./models/model1/saved/model1.p', 'rb') as handle:
             lr = pickle.load(handle)
             good = 0.0
             total = len(y_test)

             predictions = lr.predict(x_test)
             for i, instance in enumerate(x_test):
                 if predictions[i] == y_test[i]:
                      good += 1

             accuracy = good/total
             print('good: ' + str(good))
             print('total: ' + str(total))
             print('acc: ' + str(accuracy))

    if m == 'linear-tensorflow':

        a = 1
        #To do.

    if m == 'lenet':

        X_test, y_test, _ = create_input(d)
        BATCH_SIZE = 1
        #auxiliary
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, 43) #Diagonal 43x43 Identity Matrix
        #Initialize
        out = lenet(x)

        #see if the prediction is correct, create an accuracy operation
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        def evaluate(X_data, y_data, batch_size):
            """
            Evaluate how well the models does on the training set.
            """
            num_examples = len(X_data)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, batch_size):
                batch_x, batch_y = X_data[offset:offset+batch_size],y_data[offset:offset+batch_size]
                accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                total_accuracy += (accuracy * len(batch_x))
            return total_accuracy / num_examples, total_accuracy

        with tf.Session() as sess:
            #Initialize
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            #Restore saved model
            saver.restore(sess, tf.train.latest_checkpoint('./models/model3/saved/'))

            #Evaluate
            test_accuracy, total_accuracy = evaluate(X_test, y_test, BATCH_SIZE)
            print("Test Accuracy = {:.3f}".format(test_accuracy))

@click.command()
@click.option('-m', help='choose which model you want to infer with')
@click.option('-d', help='choose the directory of the images you want to infer on')
def infer(m, d):

    if m == 'linear-scikit':

          x_test, _ = create_npy_lin(d)
          #Creates the database
          images = os.listdir(d)
          #Need the images as well, x_test only contains features and not the images themselves.
          images = sorted(images)
          #They both create it in order so with sorted we are sure that the images correspond to
          #the features.
          #Load the model
          with open('./models/model1/saved/model1.p', 'rb') as handle:
             lr = pickle.load(handle)
          #Predict
          predictions = lr.predict(x_test)
          #I need the predictions in integers in order to access the classes list I defined in the
          #first lines of the code.
          pred_int = map(int, predictions)
          for i, pred in enumerate(pred_int):
              image_sorted = images[i]
              path = d + '/' + image_sorted
              img = cv2.imread(path)
              #Open image and waits some seconds to close, also can use keyboard keys to change it.
              cv2.imshow(classes[pred], img)
              cv2.waitKey(2000)
              cv2.destroyAllWindows()

    if m == 'linear-tensorflow':
        a=1
        #To do since I don't know how to save these kind of models as you read earlier

    if m == 'lenet':

        #The images I wanna infer on plus their paths for later showing the original.
        x_inf, _, image_paths = create_input(d)
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, 43) #Diagonal 43x43 Identity Matrix
        out = lenet(x)
        preds = []

        #Infer and save corresponding variables
        for i, inst in enumerate(x_inf):
            #For each image, infer and save
            feed = np.reshape(inst, (1, 32, 32, 1))
            feed_dict = {x: feed}
            prediction = tf.argmax(out, 1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint('./models/model3/saved/'))
                best = sess.run(prediction, feed_dict)
                preds.append(best[0])

        #Show the images with the corresponding inference
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            #Open image and waits some seconds to close, also can use keyboard keys to change it.
            cv2.imshow(classes[preds[i]], img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

cli.add_command(download)
cli.add_command(train)
cli.add_command(test)
cli.add_command(infer)

if __name__ == '__main__':
    cli()
