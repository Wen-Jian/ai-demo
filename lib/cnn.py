import tensorflow as tf
import numpy as np
import basic_nn_batch as bnn
import os
import cv2
import dataset as dt
import logger
import re
from shutil import copyfile

def add_cnn_layer(x_input, filter_shape, activation_function = None, strides=1):
    cnn_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape = [filter_shape[3]]))
    before_pooling = tf.nn.conv2d(x_input, cnn_filter, strides=[1,strides,strides,1], padding='SAME')  
    if (activation_function != None):
        act_input = activation_function(before_pooling)
    else:
        act_input = tf.nn.relu(before_pooling + bias)
    return act_input
    
def add_deconv_layer(x_input, filter_shape, output_shape, activation_function=None, stride = 2):
    cnn_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape = [filter_shape[3]]))
    y1 = tf.nn.conv2d_transpose(x_input,cnn_filter,output_shape,strides=[1, stride, stride, 1])
    if (activation_function == None):
        out_put = tf.nn.relu(y1)
    else:
        out_put = activation_function(y1)
    return out_put
    
def add_pooling_layer(tensor, stride = 2):
    pooling = tf.nn.max_pool(tensor, ksize=[1,2,2,1],strides=[1,stride,stride,1], padding='SAME')
    return  pooling

def train(x_input, labels, sess, input_shape, out_size, test_dataset, test_labels):
    x_shape = np.shape(x_input)
    y_shape = np.shape(labels)
    channel_size = x_shape[1]/input_shape[0]/input_shape[1]
    x_batch = tf.placeholder(tf.float32, [None, input_shape[0] * input_shape[1] * channel_size], "x_train")
    x_s = tf.reshape(x_batch, [-1, input_shape[0], input_shape[1], int(x_shape[1]/input_shape[0]/input_shape[1])])
    y_s = tf.placeholder(tf.float32, [None, y_shape[1]], "y_train")

    y1 = add_cnn_layer(x_s, [5, 5, 3, 32])
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

    for i in range(x_shape[0]//x_shape[0]):
    # for i in range(1):
        sess.run(train_step, feed_dict={x_batch: x_input[i*100: (i+1)*100]/255., y_s: labels[i*100: (i+1)*100]})
        print(sess.run(cross_entropy, feed_dict={x_batch: x_input[i*100: (i+1)*100]/255., y_s: labels[i*100: (i+1)*100]}))

    # correct_prediction = tf.equal(tf.argmax(out_put,1), tf.argmax(y_s,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x_s: test_dataset[0:10000], y_s: test_labels[0:10000]}))

def train_generator(x_input, expected_output, sess, input_shape, out_size):
    x_shape = np.shape(x_input)
    y_shape = np.shape(expected_output)
    channel_size = x_shape[1]/input_shape[0]/input_shape[1]
    
    x_batch = tf.placeholder(tf.float32, [None, input_shape[0] * input_shape[1] * channel_size], "x_train")
    batch_size = tf.shape(x_batch)[0]
    x_s = tf.reshape(x_batch, [-1, input_shape[0], input_shape[1], int(x_shape[1]/input_shape[0]/input_shape[1])])
    y_s = tf.placeholder(tf.float32, [None, y_shape[1], y_shape[2], y_shape[3]], "y_train")

    y1 = add_cnn_layer(x_s, [8, 8, 3, 32], strides=2)
    # pool_1 = add_pooling_layer(y1) # shape = [batch, 14, 14, 32]

    y2 = add_cnn_layer(y1, [8, 8, 32, 64], strides=2)
    # pool_2 = add_pooling_layer(y2) # shape = [batch, 7, 7, 64]

    deconv_1 = add_deconv_layer(y2, [8, 8, 32, 64], [batch_size, tf.shape(y1)[1], tf.shape(y1)[2], 32])

    deconv_2 = add_deconv_layer(deconv_1, [8, 8, 3, 32], [batch_size, out_size[0], out_size[1], out_size[2]], activation_function = tf.nn.sigmoid)

    output = deconv_2 * 255

    loss = tf.reduce_mean(tf.pow(output - y_s, 2))

    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    sess = tf.Session()
    saver = tf.train.Saver()

    if os.path.isfile("trained_parameters/image_generator.index"):
        saver.restore(sess, "trained_parameters/image_generator")
    else:
        sess.run(tf.global_variables_initializer())

    for i in range(1):
        sess.run(train_step, feed_dict={x_batch: x_input[0:x_shape[0]]/255., y_s: expected_output[0:x_shape[0]]})
        print(sess.run(loss, feed_dict={x_batch: x_input[0:x_shape[0]]/255., y_s: expected_output[0:x_shape[0]]}))
        if (i % 100 == 0):
            saver.save(sess, "trained_parameters/image_generator")
    #     if (i % 1000 == 0):
            # cv2.imshow('image_'+str(i),sess.run(output[0], feed_dict={x_batch: x_input[1:2]/255., y_s: expected_output[1:2]}))
    cv2.imshow('image_'+str(i),sess.run(output[0], feed_dict={x_batch: x_input[1:2]/255., y_s: expected_output[1:2]}))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pre_train(datasets, batch_size, input_shape, output_shape, channel_size, sess):

    iterator = datasets.make_one_shot_iterator()

    dataset = iterator.get_next()
    parsed_dataset = tf.parse_example(dataset, features={
            'filename': tf.FixedLenFeature([], tf.string),
            "x_image": tf.FixedLenFeature([], tf.string),
            "y_image": tf.FixedLenFeature([], tf.string)})
    # x_const_255 = tf.constant(255, dtype=tf.float32, shape=[batch_size, input_shape[0], input_shape[1],channel_size])
    x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, batch_size)], tf.float32)
    x_casted = tf.cast(x_s, tf.uint8)
    # y_const_255 = tf.constant(255, dtype=tf.float32, shape=[batch_size, output_shape[0], output_shape[1],channel_size])
    y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, batch_size)], tf.float32)
    y_casted = tf.cast(y_s, tf.uint8)
    # y1 = add_cnn_layer(x_s, [8, 8, 3, 32], strides=2)

    # y1 = add_cnn_layer(x_s, [1, 1, 3, 64], strides=1)

    # y2 = add_cnn_layer(y1, [3, 3, 32, 64], strides=2)

    # deconv_1 = add_deconv_layer(y2, [3, 3, 32, 64], [batch_size, tf.shape(y1)[1], tf.shape(y1)[2], 32], )

    # deconv_2 = add_deconv_layer(deconv_1, [8, 8, 3, 32], [batch_size, tf.shape(x_s)[1], tf.shape(x_s)[2], tf.shape(x_s)[3]])

    # deconv_3 = add_deconv_layer(deconv_2, [5, 5, 3, 3], [batch_size, output_shape[0], output_shape[1], channel_size], stride=10)

    deconv_3 = add_deconv_layer(x_s, [3, 3, 3, 3], [batch_size, output_shape[0], output_shape[1], channel_size], stride=2)
    
    # y2 = add_cnn_layer(deconv_3, [9, 9, 3, 64], strides=1)

    # y3 = add_cnn_layer(y2, [1, 1, 64, 32], strides=1)

    y4 = add_cnn_layer(deconv_3, [5, 5, 3, 3], strides=1)

    # deconv_4 = add_cnn_layer(deconv_3, [1, 1, 3, 3], strides=1)
    # deconv_5 = add_deconv_layer(deconv_4, [3, 3, 3, 3], [batch_size, output_shape[0], output_shape[1], channel_size], stride=2)

    loss = tf.reduce_mean(tf.square(y4 - y_s))

    train_step = tf.train.AdamOptimizer(1).minimize(loss)

    saver = tf.train.Saver()

    # saver.restore(sess, "trained_parameters/heigh_resolution_1_deconv_2x_500p")

    if os.path.isfile("trained_parameters/heigh_resolution_1_deconv_1conv_2x" + str(batch_size) + "p.index"):
        saver.restore(sess, "trained_parameters/heigh_resolution_1_deconv_1conv_2x" + str(batch_size) + "p")
    else:
        sess.run(tf.global_variables_initializer())
    
    # output = np.array(sess.run(tf.cast(deconv_4, tf.uint8)))
    # print(output)
    # output = np.array(sess.run(tf.cast(deconv_4, tf.uint8)))
    # cv2.imshow('image_test', output[0])
    # cv2.imshow('image_test_origin', np.array(sess.run(tf.cast(x_s * 255, tf.uint8), feed_dict={x_s: x_image})[0]))
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    while True:
        sess.run(train_step)
        saver.save(sess, "trained_parameters/heigh_resolution_1_deconv_1conv_2x_" + str(batch_size) + "p")
        loss_val = sess.run(loss)
        print(loss_val)
        # logger.logger(str(loss_val))
        # try:
        #     sess.run(train_step)
        #     saver.save(sess, "trained_parameters/heigh_resolution_1_deconv_" + str(batch_size) + "p")
        #     print(sess.run(loss))
        # except:
        #     print("結束了")
        #     break

def heigh_resolution_generator_train_with_imgs(x_img_data, y_img_data, batch_size, input_shape, output_shape, channel_size):

    x_datas = tf.cast(tf.stack(x_img_data), tf.float32)
    batch_size = np.shape(x_datas)[0]
    x_s = tf.reshape(x_datas, [batch_size, input_shape[0], input_shape[1], channel_size])
    y_datas = tf.cast(tf.stack(y_img_data), tf.float32)
    y_s = tf.reshape(y_datas, [batch_size, output_shape[0], output_shape[1], channel_size])
    
    y1 = add_cnn_layer(tf.divide(x_s, 255), [8, 8, 3, 32], strides=2)

    y2 = add_cnn_layer(y1, [3, 3, 32, 64], strides=2)

    deconv_1 = add_deconv_layer(y2, [3, 3, 32, 64], [batch_size, tf.shape(y1)[1], tf.shape(y1)[2], 32], )

    deconv_2 = add_deconv_layer(deconv_1, [8, 8, 3, 32], [batch_size, tf.shape(x_s)[1], tf.shape(x_s)[2], tf.shape(x_s)[3]])

    deconv_3 = add_deconv_layer(deconv_2, [5, 5, 3, 3], [batch_size, output_shape[0], output_shape[1], channel_size], stride=10)

    output = deconv_3 * 255

    loss = tf.reduce_mean(tf.square(output - tf.cast(y_s, tf.float32)))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    sess = tf.Session()
    saver = tf.train.Saver()

    if os.path.isfile("trained_parameters/heigh_resolution_" + str(batch_size) + "p.index"):
        saver.restore(sess, "trained_parameters/heigh_resolution_"+ str(batch_size) +"p")
    else:
        sess.run(tf.global_variables_initializer())
    
    sess.run(train_step)
    print(sess.run(loss))
    saver.save(sess, "trained_parameters/heigh_resolution_"+ str(batch_size) +"p")

def img_generator_train_with_imgs(datasets, batch_size, input_shape, output_shape, channel_size, sess):

    iterator = datasets.make_one_shot_iterator()

    dataset = iterator.get_next()
    parsed_dataset = tf.parse_example(dataset, features={
            'filename': tf.FixedLenFeature([], tf.string),
            "x_image": tf.FixedLenFeature([], tf.string),
            "y_image": tf.FixedLenFeature([], tf.string)})
    x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, batch_size)], tf.float32)
    x_casted = tf.cast(x_s, tf.uint8)
    y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, batch_size)], tf.float32)
    y_casted = tf.cast(y_s, tf.uint8)
    y1 = add_cnn_layer(tf.div(x_s, tf.constant(255, dtype=tf.float32, shape=[batch_size, input_shape[0], input_shape[1],channel_size])), [8, 8, 3, 32], strides=2)

    y2 = add_cnn_layer(y1, [3, 3, 32, 64], strides=2)

    deconv_1 = add_deconv_layer(y2, [3, 3, 32, 64], [batch_size, tf.shape(y1)[1], tf.shape(y1)[2], 32], )

    deconv_2 = add_deconv_layer(deconv_1, [8, 8, 3, 32], [batch_size, tf.shape(x_s)[1], tf.shape(x_s)[2], tf.shape(x_s)[3]])

    # deconv_3 = add_deconv_layer(deconv_2, [5, 5, 3, 3], [batch_size, output_shape[0], output_shape[1], channel_size], stride=10)

    output = deconv_2 * 255

    loss = tf.reduce_mean(tf.square(output - tf.cast(y_s, tf.float32)))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # # sess = tf.Session()
    saver = tf.train.Saver()

    if os.path.isfile("trained_parameters/normal_generator_" + str(batch_size) + "p.index"):
        saver.restore(sess, "trained_parameters/normal_generator_" + str(batch_size) + "p")
    else:
        sess.run(tf.global_variables_initializer())
    
    while True:
        try:
            sess.run(train_step)
            saver.save(sess, "trained_parameters/normal_generator_" + str(batch_size) + "p")
            print(sess.run(loss))
        except:
            print("結束了")
            break

def generate_image(x_image, batch_size, input_shape, output_shape, channel_size, sess, parameter_name):

    x_s = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], channel_size], "x_test")
  
    # y1 = add_cnn_layer(x_s, [8, 8, 3, 32], strides=2)

    # y2 = add_cnn_layer(y1, [3, 3, 32, 64], strides=2)

    # deconv_1 = add_deconv_layer(y2, [3, 3, 32, 64], [batch_size, tf.shape(y1)[1], tf.shape(y1)[2], 32], )

    # deconv_2 = add_deconv_layer(deconv_1, [8, 8, 3, 32], [batch_size, tf.shape(x_s)[1], tf.shape(x_s)[2], tf.shape(x_s)[3]])

    deconv_3 = add_deconv_layer(x_s, [3, 3, 3, 3], [batch_size, output_shape[0], output_shape[1], channel_size], stride=2)

    avg_pooling = tf.nn.avg_pool(deconv_3, ksize=[1,2,2,1],strides=[1,1,1,1], padding='SAME')

    max_pooling = tf.nn.max_pool(deconv_3, ksize=[1,2,2,1],strides=[1,1,1,1], padding='SAME')

    output = tf.cast((deconv_3 + avg_pooling * 0.6), tf.uint8)

    # sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, "trained_parameters/" + parameter_name)
    # cv2.imshow('image_test', np.array(sess.run(output, feed_dict={x_s: x_image})[0]))
    # cv2.imshow('origin', np.array(sess.run(tf.cast(x_s, tf.uint8), feed_dict={x_s: x_image})[0]))
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return np.array(sess.run(output, feed_dict={x_s: x_image})[0])

def train_heigh_resolution_generator(datasets, batch_size, input_shape, output_shape, channel_size, sess):
    iterator = datasets.make_one_shot_iterator()

    dataset = iterator.get_next()
    parsed_dataset = tf.parse_example(dataset, features={
            'filename': tf.FixedLenFeature([], tf.string),
            "x_image": tf.FixedLenFeature([], tf.string),
            "y_image": tf.FixedLenFeature([], tf.string)})
    x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, batch_size)], tf.float32)
    y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, batch_size)], tf.float32)

    # deconv_1 = add_deconv_layer(x_s, [3, 3, 3, 3], [batch_size, output_shape[0], output_shape[1], channel_size], stride=2)
    cnn_filter_1 = tf.Variable(tf.truncated_normal([3, 3, 16, 3], stddev=0.1))
    bias_1 = tf.Variable(tf.constant(0.1, shape = [3]))
    y1 = tf.nn.conv2d_transpose(x_s, cnn_filter_1,[batch_size, output_shape[0], output_shape[1], 16],strides=[1, 2, 2, 1])
    deconv_1 = tf.nn.relu(y1)




    cnn_filter = tf.Variable(tf.truncated_normal([5,5,16,3], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape = [3]))
    before_pooling = tf.nn.conv2d(deconv_1, cnn_filter, strides=[1,1,1,1], padding='SAME') 

    loss = tf.reduce_mean(tf.square(before_pooling - y_s))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    if os.path.isfile("trained_parameters/heigh_resolution_1_deconv_32f_1conv_2x" + str(batch_size) + "p.index"):
        # saver = tf.train.Saver()
        # saver.restore(sess, "trained_parameters/heigh_resolution_1_deconv_1conv_2x" + str(batch_size) + "p")

        saver = tf.train.Saver()
        saver.restore(sess, "trained_parameters/heigh_resolution_1_deconv_32f_1conv_2x" + str(batch_size) + "p")
    else:
        # saver = tf.train.Saver([cnn_filter_1, bias_1])
        # sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "trained_parameters/finished_params/heigh_resolution_1_deconv_2x_300p")
        # sess.run(tf.variables_initializer(
        #     [cnn_filter, bias]))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
    while True:
        sess.run(train_step)
        saver.save(sess, "trained_parameters/heigh_resolution_1_deconv_32f_1conv_2x_" + str(batch_size) + "p")
        loss_val = sess.run(loss)
        print(loss_val)

def train_heigh_resolution(datasets, batch_size, input_shape, output_shape, channel_size, sess):
    count = 0
    iterator = datasets.make_one_shot_iterator()
    dataset = iterator.get_next()
    parsed_dataset = tf.parse_example(dataset, features={
            'filename': tf.FixedLenFeature([], tf.string),
            "x_image": tf.FixedLenFeature([], tf.string),
            "y_image": tf.FixedLenFeature([], tf.string)})
    x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, batch_size)], tf.float32)
    y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, batch_size)], tf.float32)
    
    # deconv_3 = add_deconv_layer(x_s, [3, 3, 32, 3], [batch_size, input_shape[0] * 2, input_shape[1] * 2, 32], stride=2)

    # 七層CNN 同尺寸 低解 -> 高解
    # y1 = add_cnn_layer(x_s, [3, 3, 3, 16], strides=1)

    # y2 = add_cnn_layer(y1, [3, 3, 16, 16], strides=1)
    # y3 = add_cnn_layer(y2, [3, 3, 16, 16], strides=1)

    # pooled_y3 = add_pooling_layer(y3, 1)

    # y4 = add_cnn_layer(pooled_y3, [3, 3, 16, 16], strides=1)

    # y5 = add_cnn_layer(y4, [3, 3, 16, 16], strides=1)

    # y6 = y5 + pooled_y3

    # y7 = add_cnn_layer(y6, [3, 3, 16, 32], strides=1)

    # y8 = add_cnn_layer(y7, [3, 3, 32, 32], strides=1)

    # y9 = y8 + add_cnn_layer(y6, [1, 1, 16, 32], strides=1)

    # y7 = add_cnn_layer(y9, [3,3,32,3], strides=1) + x_s

    # pred = tf.nn.avg_pool(y7, ksize=[1,2,2,1],strides=[1,1,1,1], padding='SAME')
    # origin_loss = tf.reduce_mean(tf.square(x_s - y_s))
    # loss = tf.reduce_mean(tf.square(pred - y_s))

    # up scaling down sampling
    # conv1 = tf.nn.conv2d(x_s, filter=tf.Variable(tf.truncated_normal([3,3,3,3], stddev=0.1)),strides=[1,2,2,1],padding='SAME')
    # bias = tf.Variable(tf.constant(0.1, shape = [filter_shape[3]]))
    # conv1 = tf.nn.relu(before_pooling + bias)


    conv1 = add_cnn_layer(x_s, [3, 3, 3, 3], strides=2)
    
    conv1 = add_cnn_layer(conv1, [360, 640, 3, 3], strides=1)

    # pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,1,1,1], padding='SAME')

    loss = tf.reduce_mean(tf.square(conv1 - y_s))
    
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(loss)
    # grad_and_var = optimizer.compute_gradients(loss)
    varis = tf.trainable_variables()
    # gradients = tf.gradients(loss, tf.trainable_variables())
    gradients = optimizer.compute_gradients(loss)
    # count = 0

    if os.path.isfile("trained_parameters/heigh_resolution_large_to_small_with_fv_" + str(batch_size) + "p.index"):
        saver = tf.train.Saver()
        saver.restore(sess, "trained_parameters/heigh_resolution_large_to_small_with_fv_" + str(batch_size) + "p")
    else:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
    # dest = '/Users/wen/Desktop/programing/AI_demo/image_generator_train/smaller_100/'
    while True:
        count += 1
        _, loss_val, grad = sess.run([train_step, loss, gradients])
        saver.save(sess, "trained_parameters/heigh_resolution_large_to_small_with_fv_" + str(batch_size) + "p")
        grad = [gradient for gradient, varis in grad]
        # file_path = file_path.decode("utf-8")
        # if loss_val < 100:
        #     count += 1
        #     print(loss_val)
        #     print(file_path)
        #     file_name = re.search('\d*\.jpg', file_path).group()
        #     print(file_name)
        #     copyfile(file_path, dest + 'x_360p/' + file_name)
        #     copyfile('/Users/wen/Desktop/programing/AI_demo/image_generator_train/height_360p/' + file_name.group(), dest + 'height_360p/' + file_name.group())
        #     # if os.path.isfile("/Users/wen/Desktop/programing/AI_demo/image_generator_train/y/" + file_name.group()):
        #     #     file_path = "/Users/wen/Desktop/programing/AI_demo/image_generator_train/y/" + file_name.group()
        #     #     copyfile(file_path, dest + '/1920 x 1080/' + file_name.group())
        #     #     os.remove(file_path)
        #     #     os.remove("/Users/wen/Desktop/programing/AI_demo/image_generator_train/x_360p/" + file_name.group())
        #     #     os.remove("/Users/wen/Desktop/programing/AI_demo/image_generator_train/height_360p/" + file_name.group())
        #     #     os.remove("/Users/wen/Desktop/programing/AI_demo/image_generator_train/720p/" + file_name.group())

        #     # elif os.path.isfile("/Users/wen/Desktop/programing/AI_demo/image_generator_train/rename_x/" + file_name.group()):
        #     #     file_path = "/Users/wen/Desktop/programing/AI_demo/image_generator_train/rename_x/" + file_name.group()
        #     #     copyfile(file_path, dest + '/' + file_name.group())
        #     #     os.remove(file_path)
        #     #     os.remove("/Users/wen/Desktop/programing/AI_demo/image_generator_train/x_360p/" + file_name.group())
        #     #     os.remove("/Users/wen/Desktop/programing/AI_demo/image_generator_train/height_360p/" + file_name.group())
        #     #     # os.remove("/Users/wen/Desktop/programing/AI_demo/image_generator_train/720p/" + file_name.group())

        #     # elif os.path.isfile("/Users/wen/Desktop/programing/AI_demo/image_generator_train/720p/" + file_name.group()):
        #     #     file_path = "/Users/wen/Desktop/programing/AI_demo/image_generator_train/720p/" + file_name.group()
        #     #     copyfile(file_path, dest + '/' + file_name.group())
        #     #     os.remove(file_path)
        #     #     os.remove("/Users/wen/Desktop/programing/AI_demo/image_generator_train/x_360p/" + file_name.group())
        #     #     os.remove("/Users/wen/Desktop/programing/AI_demo/image_generator_train/height_360p/" + file_name.group())
        #     print(count)
        print(loss_val)
        # print(grad)
        # print(ori_loss)

def generate_2_layer_image(x_image, batch_size, input_shape, output_shape, channel_size, sess, parameter_name):
    x_s = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], channel_size], "x_test")
  
    deconv_3 = add_deconv_layer(x_s, [3, 3, 3, 3], [batch_size, output_shape[0], output_shape[1], channel_size], stride=2)

    cnn_filter = tf.Variable(tf.truncated_normal([5,5,3,3], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape = [3]))
    before_pooling = tf.nn.conv2d(deconv_3, cnn_filter, strides=[1,1,1,1], padding='SAME') 


    output = tf.cast(before_pooling, tf.uint8)

    # sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, "trained_parameters/" + parameter_name)
    cv2.imshow('image_test', np.array(sess.run(output, feed_dict={x_s: x_image})[0]))
    cv2.imshow('origin', np.array(sess.run(tf.cast(x_s, tf.uint8), feed_dict={x_s: x_image})[0]))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return np.array(sess.run(output, feed_dict={x_s: x_image})[0])

def generate_height_image(x_image, batch_size, input_shape, output_shape, channel_size, sess, parameter_name):
    x_s = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], channel_size], "x_test")
  
    # y1 = add_cnn_layer(x_s, [9, 9, 3, 32], strides=1)

    y2 = add_cnn_layer(x_s, [3, 3, 3, 3], strides=2)

    pred = y2

    output = tf.cast(pred, tf.uint8)

    # sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, "trained_parameters/" + parameter_name)
    # cv2.imshow('image_test', np.array(sess.run(output, feed_dict={x_s: x_image})[0]))
    # cv2.imshow('origin', np.array(sess.run(tf.cast(x_s, tf.uint8), feed_dict={x_s: x_image})[0]))
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return np.array(sess.run(output, feed_dict={x_s: x_image})[0])