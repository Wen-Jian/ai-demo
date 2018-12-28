import tensorflow as tf
import numpy as np
import os
import sys
import glob
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import create_mnist_jpg as im_creator
from tensorflow.examples.tutorials.mnist import input_data
import basic_nn_batch as batch_nn
import cnn 
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_img_file_path = os.path.join(os.getcwd(),'MnistImage/Train')
dataset = im_creator.img_to_data_set(train_img_file_path)
traning_lables = im_creator.train_labels()

# print(np.shape(dataset[0]))
# print(np.reshape(dataset[0], [1, 28, 28, 3]))
    
# shape of dateset = [55000, 28*28*3]
# shape of labes = [55000, 10]
# input_shape = np.shape(dataset[0])
# shape_size = input_shape[0] * input_shape[1] * input_shape[2]
# dataset = np.reshape(dataset, [np.shape(dataset)[0], shape_size] )
# out_size = np.shape(traning_lables)[1]
sess = tf.Session()


test_img_file_path = os.path.join(os.getcwd(),'MnistImage/Test')
test_dataset = im_creator.img_to_data_set(test_img_file_path)
# input_shape = np.shape(test_dataset[0])
# shape_size = input_shape[0] * input_shape[1] * input_shape[2]
# test_dataset = np.reshape(test_dataset, [np.shape(test_dataset)[0], shape_size] )
test_labels = im_creator.test_labels()

# basic nn
# batch_nn.predic_and_train(dataset, traning_lables, shape_size, out_size, sess, test_dataset, test_labels)

# convolution network
cnn.train(np.array(dataset, "f"), traning_lables, sess, 28, 10, 10, test_dataset, test_labels)

