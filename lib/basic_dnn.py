import tensorflow as tf
import numpy as np
import cv2
from tensorflow.python import debug as tf_debug 
import time

NearZero = 1e-10
def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(initial_value=tf.random_normal(shape=[in_size, out_size], dtype=tf.float32), dtype=tf.float32)
    biases = tf.Variable(initial_value=(tf.zeros([1, out_size], dtype=tf.float32) + 0.1), dtype=tf.float32)
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    
    return outputs

def train(x_data, y_data):
    print(np.shape(x_data))
    xs = tf.placeholder(tf.float32, shape=[None, 28*28*3], name='xs')
    ys = tf.placeholder(tf.float32, shape=[None, 10,], name='ys')

    weights_1 = tf.Variable(initial_value=tf.random_normal(shape=[28*28*3, 10], dtype=tf.float32), dtype=tf.float32)
    biases_1 = tf.Variable(initial_value=(tf.zeros([1, 10], dtype=tf.float32) + 0.1), dtype=tf.float32)
    Wx_plus_b = tf.matmul(xs, weights_1) + biases_1

    weights_2 = tf.Variable(initial_value=tf.random_normal(shape=[10, 10], dtype=tf.float32), dtype=tf.float32)
    biases_2 = tf.Variable(initial_value=(tf.zeros([1, 10], dtype=tf.float32) + 0.1), dtype=tf.float32)
    Wx_plus_b_2 = tf.matmul(Wx_plus_b, weights_2) + biases_2
    out = tf.nn.softmax(Wx_plus_b_2) 

    #layer 1
    # W1 = tf.get_variable('w1', [28*28*3, 100], initializer=tf.random_normal_initializer())
    # b1 = tf.get_variable('b1', [1,], initializer=tf.random_normal_initializer())
    # y1 = tf.nn.sigmoid(tf.matmul(x_data, W1) + b1) 

    # #layer 2
    # W2 = tf.get_variable('w2',[100,10], initializer= 
    # tf.random_normal_initializer())
    # b2 = tf.get_variable('b2',[1,], initializer=tf.random_normal_initializer())
    # y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

    # #output
    # y = y2

    # loss = -tf.reduce_sum(ys * tf.log(y2))

    # train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # hidden_layer_1 = add_layer(xs, 28*28*3, 10, tf.nn.softmax)

    # hidden_layer_2 = add_layer(xs, 28*28*3, 10, tf.nn.softmax)

    loss = -tf.reduce_sum(ys * tf.log(out))
    
    # equality = tf.equal(tf.argmax(hidden_layer_1, 1), tf.argmax(ys, 1))

    # accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    grad_and_var = optimizer.compute_gradients(loss)

    gradient = [grad for grad, var in grad_and_var]

    train_step = optimizer.apply_gradients(grad_and_var)

    # saver = tf.train.Saver()

    sess = tf.Session()
    # # sess = tf_debug.LocalCLIDebugWrapperSession(sess) 
    sess.run(tf.global_variables_initializer())
    # dimension = len(x_data)
    # localtime = time.asctime( time.localtime(time.time()) )

    for j in range(4):
        # training
        # sess.run(x_data[j], feed_dict={xs: x_data[j], ys: y_data[j]})
            
        # to see the step improvement
        # print(sess.run(accuracy, feed_dict={xs: x_data[j], ys: y_data[j]}))
        # print(sess.run(tf.matmul(xs, tf.reshape(weights[:, 0], [28*28*3, 1])), feed_dict={xs: x_data[j], ys: y_data[j]}))
        print(sess.run(loss, feed_dict={xs: x_data[j], ys: y_data[j]}))
        # print(sess.run(weights_1, feed_dict={xs: x_data[i], ys: y_data[i]}))
    # print(sess.run(grad_and_var, feed_dict={xs: x_data, ys: y_data}))
    # print(sess.run(hidden_layer_2, feed_dict={xs: x_data, ys: y_data}))

    
    # saver.save(sess, "trained_parameters/%s.ckpt" % localtime)

        