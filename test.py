import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import create_mnist_jpg as im_creator
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])

#layer 1
W1 = tf.get_variable('w1', [784, 100], initializer=tf.random_normal_initializer())
b1 = tf.get_variable('b1', [1,], initializer=tf.random_normal_initializer())
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1) 

#layer 2
W2 = tf.get_variable('w2',[100,10], initializer = tf.random_normal_initializer())
b2 = tf.get_variable('b2',[1,], initializer=tf.random_normal_initializer())
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

#output
y = y2
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 
reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train_img_file_path = os.path.join(os.getcwd(),'MnistImage/Train')
# dataset = im_creator.img_to_data_set(train_img_file_path)
# x = np.reshape(dataset[0:100], [100, 28*28*3])
# print(np.shape(dataset[0:100]))

# for i in range((len(dataset)//100)):
#   batch_xs, batch_ys = mnist.train.next_batch(100)
#   print(np.shape(batch_xs))
#   # sess.run(train_step, feed_dict={x: dataset[i*100:(i+1)*100], y_: batch_ys})
#   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))

test_img_file_path = os.path.join(os.getcwd(),'MnistImage/Test')
test_dataset = im_creator.img_to_data_set(test_img_file_path)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))