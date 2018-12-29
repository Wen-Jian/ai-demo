import tensorflow as tf
import numpy as np
import basic_nn_batch as bnn

def add_cnn_layer(x_input, batch_size, filter_shape, activation_function = None):
    cnn_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape = [filter_shape[3]]))
    before_pooling = tf.nn.conv2d(x_input, cnn_filter, strides=[1,1,1,1], padding='SAME')  
    if (activation_function != None):
        act_input = activation_function(before_pooling)
    else:
        act_input = tf.nn.relu(before_pooling + bias)
    return act_input
    
    
def add_pooling_layer(tensor):
    pooling = tf.nn.max_pool(tensor, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    return  pooling

def train(x_input, labels, sess, input_shape, out_size, batch_size, test_dataset, test_labels):
    x_shape = np.shape(x_input)
    y_shape = np.shape(labels)
    channel_size = x_shape[1]/input_shape[0]/input_shape[1]
    x_batch = tf.placeholder(tf.float32, [None, input_shape[0] * input_shape[1] * channel_size], "x_train")
    x_s = tf.reshape(x_batch, [-1, input_shape[0], input_shape[1], int(x_shape[1]/input_shape[0]/input_shape[1])])
    y_s = tf.placeholder(tf.float32, [None, y_shape[1]], "y_train")

    y1 = add_cnn_layer(x_s, tf.shape(x_s)[0], [5, 5, 3, 32])
    pool_1 = add_pooling_layer(y1)
    y2 = add_cnn_layer(pool_1, tf.shape(pool_1)[0], [3, 3, 32, 64])
    after_pooling = add_pooling_layer(y2)

    single_shape = int((input_shape[0]/4) * (input_shape[1]/4) * 64)
    y3 = tf.reshape(tensor=after_pooling, shape=[-1, single_shape])
    bnn_1 = bnn.add_lyaer(y3, single_shape, 1024, tf.nn.relu)
    out_put = bnn.add_lyaer(bnn_1, 1024, out_size, tf.nn.softmax)
    

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_s * tf.log(out_put), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # grad_and_var = tf.train.AdamOptimizer(1e-4).compute_gradients(cross_entropy)

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # for i in range(1):
    # # batch_xs, batch_ys = mnist.train.next_batch(100)
    # # sess.run(train_step, feed_dict={x_s: batch_xs, y_s: batch_ys, keep_prob: 0.5})
    #     sess.run(train_step, feed_dict={x_s: x_input[i*100: (i+1)*100]/255., y_s: labels[i*100: (i+1)*100]})
    #     print(sess.run(cross_entropy, feed_dict={x_s: x_input[i*100: (i+1)*100]/255., y_s: labels[i*100: (i+1)*100]}))

    for i in range(x_shape[0]//batch_size):
    # for i in range(1):
        sess.run(train_step, feed_dict={x_batch: x_input[i*100: (i+1)*100]/255., y_s: labels[i*100: (i+1)*100]})
        print(sess.run(cross_entropy, feed_dict={x_batch: x_input[i*100: (i+1)*100]/255., y_s: labels[i*100: (i+1)*100]}))

    # correct_prediction = tf.equal(tf.argmax(out_put,1), tf.argmax(y_s,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x_s: test_dataset[0:10000], y_s: test_labels[0:10000]}))