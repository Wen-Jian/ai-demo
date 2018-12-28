import os
import sys
import glob
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import create_mnist_jpg as im_creator
import tensorflow as tf
import numpy as np
import cnn 
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    # y_pre = sess.run(prediction, feed_dict={x_s: v_xs, y_s: v_ys, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_s,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x_s: v_xs, y_s: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


train_img_file_path = os.path.join(os.getcwd(),'MnistImage/Train')
dataset = im_creator.img_to_data_set(train_img_file_path)
traning_lables = im_creator.train_labels()

input_shape = np.shape(dataset[0])
shape_size = input_shape[0] * input_shape[1] * input_shape[2]
dataset = np.reshape(dataset, [np.shape(dataset)[0], shape_size] )
out_size = np.shape(traning_lables)[1]

test_img_file_path = os.path.join(os.getcwd(),'MnistImage/Test')
test_dataset = im_creator.img_to_data_set(test_img_file_path)
input_shape = np.shape(test_dataset[0])
shape_size = input_shape[0] * input_shape[1] * input_shape[2]
test_dataset = np.reshape(test_dataset, [np.shape(test_dataset)[0], shape_size] )
test_labels = im_creator.test_labels()

# define placeholder for inputs to network
x_s = tf.placeholder(tf.float32, [None, 2352])   # 28x28
y_s = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x_s, [-1, 28, 28, 3])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
# W_conv1 = weight_variable([5,5, 3,32]) # patch 5x5, in size 1, out size 32
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
# h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32
y1 = cnn.add_cnn_layer(x_image, tf.shape(x_image)[0], [5, 5, 3, 32])
h_pool1 = cnn.add_pooling_layer(y1)

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_s * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# grad_and_var = tf.train.AdamOptimizer(0.2).compute_gradients(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
# batch_xs, batch_ys = mnist.train.next_batch(100)
# print(sess.run([grad[0] for grad in grad_and_var], feed_dict={x_s: dataset[0: 1]/255., y_s: traning_lables[0: 1], keep_prob: 0.5}))
# print(sess.run(cross_entropy, feed_dict={x_s: dataset[0: 1]/255., y_s: traning_lables[0: 1], keep_prob: 0.5}))


for i in range(500):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    # sess.run(train_step, feed_dict={x_s: batch_xs, y_s: batch_ys, keep_prob: 0.5})
    # print(sess.run(cross_entropy, feed_dict={x_s: batch_xs, y_s: batch_ys, keep_prob: 0.5}))
    sess.run(train_step, feed_dict={x_s: dataset[i*100: (i+1)*100]/255., y_s: traning_lables[i*100: (i+1)*100], keep_prob: 0.5})
    print(sess.run(cross_entropy, feed_dict={x_s: dataset[i*100: (i+1)*100]/255., y_s: traning_lables[i*100: (i+1)*100], keep_prob: 0.5}))
    # if i % 50 == 0:
    #     print("=================================")
    #     print(compute_accuracy(
    #         mnist.test.images[:1000], mnist.test.labels[:1000]))
    #     print("=================================")

    if i % 50 == 0:
        print("=================================")
        print(compute_accuracy(test_dataset[1:1000], test_labels[1:1000]))
        print("=================================", np.shape(test_labels[0:1000]))
        