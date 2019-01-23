import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import glob
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import create_mnist_jpg as im_creator
import basic_nn_batch as batch_nn
import cnn 
import tensorflow as tf

train_x_file_path = os.path.join(os.getcwd(),'image_generator_train/x_samll')
train_y_file_path = os.path.join(os.getcwd(),'image_generator_train/y')
dataset = im_creator.img_to_data_set(train_x_file_path)
input_shape = np.shape(dataset[0])
shape_size = input_shape[0] * input_shape[1] * input_shape[2]
batch_size = np.shape(dataset)[0]
dataset = np.reshape(dataset, [batch_size, shape_size] )

expected_output = im_creator.img_to_data_set(train_y_file_path)
expected_output_shape = np.shape(expected_output[0])
# expected_output_shape_size = expected_output_shape[0] * expected_output_shape[1] * expected_output_shape[2]
# expected_output = np.reshape(expected_output, [np.shape(expected_output)[0], expected_output_shape_size] )

sess = tf.Session()
cnn.train_heigh_resolution_generator(dataset[0:batch_size], expected_output[0:batch_size], sess, [input_shape[0], input_shape[1]], expected_output_shape)
