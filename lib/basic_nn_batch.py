import tensorflow as tf
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import create_mnist_jpg as im_creator
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def add_lyaer(input, input_size, out_size, activation_fucation):
    # W1 = tf.get_variable('w1', [input_size, out_size], initializer=tf.random_normal_initializer())
    with tf.name_scope('basic_dnn_layer'):
        with tf.name_scope('weight'):
            W1 = tf.Variable(initial_value=tf.random_normal(shape=[input_size, out_size], dtype=tf.float32), dtype=tf.float32, name='weights')
        with tf.name_scope('bias'):
            b1 = tf.Variable(initial_value=tf.random_normal(shape=[1, ], dtype=tf.float32), dtype=tf.float32, name='bias')
        with tf.name_scope('activation_function'):
            if activation_fucation == None:
                y1 = tf.nn.sigmoid(tf.matmul(input, W1) + b1) 
            else:
                y1 = activation_fucation(tf.matmul(input, W1) + b1) 
            return y1

def train(dataset, batch_ys, in_size, out_size):
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, in_size], name="x_input")
        y_ = tf.placeholder(tf.float32, [None, out_size], "y_input")
    y1 = add_lyaer(x, in_size, 100, None)
    y2 = add_lyaer(y1, 100, out_size, tf.nn.softmax)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2), 
        reduction_indices=[1]))
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)

    for i in range(10):
        for i in range(len(dataset)//100):
            # batch_xs, batch_y = batch_ys.next_batch(1)
            sess.run(train_step, feed_dict={x: dataset[i*100:(i+1)*100], y_: batch_ys[i*100:(i+1)*100]})
            if i%100 == 0:
                print(sess.run(cross_entropy, feed_dict={x: dataset[i:(i+1)*100], y_: batch_ys[i:(i+1)*100]}))
    
    test_img_file_path = os.path.join(os.getcwd(),'MnistImage/Test')
    test_dataset = im_creator.img_to_data_set(test_img_file_path)
    correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(np.shape(test_dataset))
    print(sess.run(accuracy, feed_dict={x: test_dataset[0:10000], y_: im_creator.test_labels()[0:10000]}))