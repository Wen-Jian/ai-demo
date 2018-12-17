import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python import debug as tf_debug 
import time

NearZero = 1e-10
def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(initial_value=tf.random_normal(shape=[in_size, out_size], dtype=tf.float32), dtype=tf.float32)
    biases = tf.Variable(initial_value=(tf.zeros([1, ], dtype=tf.float32) + 0.1), dtype=tf.float32)
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    
    return outputs

def train(x_data, y_data):
    xs = tf.placeholder(tf.float32, shape=[None, 28*28*3], name='xs')
    ys = tf.placeholder(tf.float32, shape=[None, 10], name='ys')


    hidden_layer_1 = add_layer(xs, 28*28*3, 10, tf.nn.softmax)

    hidden_layer_2 = add_layer(hidden_layer_1, 10, 10, tf.nn.softmax)

    loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(hidden_layer_2)))
    
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    grad_and_var = optimizer.compute_gradients(loss)

    gradient = [grad for grad, var in grad_and_var]

    train_step = optimizer.apply_gradients(grad_and_var)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for j in range(10):
        # training
        sess.run(train_step, feed_dict={xs: np.reshape(x_data[j], [1, 28*28*3]), ys: y_data[j:j+1]})
            
    # print(sess.run(grad_and_var, feed_dict={xs: x_data, ys: y_data}))
    # print(sess.run(hidden_layer_2, feed_dict={xs: x_data, ys: y_data}))

    test_img_file_path = os.path.join(os.getcwd(),'MnistImage/Test')
    test_dataset = im_creator.img_to_data_set(test_img_file_path)
    correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: test_dataset, y_: 
    mnist.test.labels}))

        