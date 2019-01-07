
import tensorflow as tf
import cv2
import numpy as np
import scipy.misc
import os
import glob
import sys
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

def x_train_set_to_jpg():
    path = os.path.join(os.getcwd(),'MnistImage/Train')
    if  os.path.isdir(path) != True:
        os.mkdir(path)
    count = 1
    for e in x_train:
        img = np.array(e, dtype='float').reshape(28,28)
        scipy.misc.imsave(
            os.path.join(path,'x_train%05d.jpg' %(count)),
            img)
        count += 1
        print('x_train%05d.jpg is saved' %(count))

def x_test_set_to_jpg():
    count = 1
    path = os.path.join(os.getcwd(),'MnistImage/Test')
    if  os.path.isdir(path) != True:
        os.mkdir(path)
    for e in x_test:
        img = np.array(e, dtype='float').reshape(28,28)
        scipy.misc.imsave(
            os.path.join(path,'x_test%05d.jpg' %(count)),
            img)
        count += 1
        print('x_test%05d.jpg is saved' %(count))

def img_to_data_set(files_path = None):

    if files_path == None:
        files_path = os.path.join(os.getcwd(),'MnistImage/Train')
    
    filelist = glob.glob(os.path.join(files_path, '*'))
    filelist.sort()
    images = []
    count = 0
    for file_name in filelist:
        process_rate = float(count/len(filelist)) * 100
        images.append(_parse_function(file_name))
        print(np.shape(images))
        sys.stdout.write("\rDoing thing %i ％" % process_rate)
        sys.stdout.flush()
        count += 1
    
    # dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return images

def _parse_function(filename):
    image_np_array = cv2.imread(filename)
    
    return image_np_array

def create_all_jpg_set():
    x_train_set_to_jpg()
    x_test_set_to_jpg()

def train_labels():
    lable_array = []
    for i in range(len(y_train)):
        lable_array.append(y_train[i])
    return lable_array

def test_labels():
    lable_array = []
    for i in range(len(y_test)):
        lable_array.append(y_test[i])
    return lable_array
# print(x_train[0].shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print("---")

if __name__ == '__main__':
    x_train_set_to_jpg()
    x_test_set_to_jpg()