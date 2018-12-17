import tensorflow as tf
import numpy as np
import os
import sys
import glob
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import create_mnist_jpg as im_creator
from tensorflow.examples.tutorials.mnist import input_data
import basic_nn_batch as batch_nn
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_img_file_path = os.path.join(os.getcwd(),'MnistImage/Train')
dataset = im_creator.img_to_data_set(train_img_file_path)
traning_lables = im_creator.train_labels()

print(np.shape(dataset))
print(np.reshape(dataset[0], [1, 28, 28, 3]))
    
# shape of dateset = [55000, 28*28*3]
# shape of labes = [55000, 10]
# for i in range(10):
#     batch_nn.train(dataset, traning_lables, np.shape(dataset)[1], np.shape(traning_lables)[1])