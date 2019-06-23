import tensorflow as tf
import numpy as np
import basic_nn_batch as bnn
import os
import cv2
import dataset as dt
import cnn

def cnn_with_batch_norm(x_s, k_size, input_features, output_features, strides):
    y1 = cnn.add_cnn_layer(x_s, [k_size[0], k_size[1], input_features, output_features], strides=strides)
    mean, variance = tf.nn.moments(y1, [0,1,2] if input_features > 1 else [0])
    A_1_bn = tf.nn.batch_normalization(y1, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-4)
    y1_act = tf.nn.relu(A_1_bn)
    A_1_pool = tf.nn.max_pool2d(y1_act, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')
    return A_1_pool

def train_heigh_resolution_with_gpu(datasets, batch_size, input_shape, output_shape, channel_size, sess):
    count = 0
    iterator = tf.compat.v1.data.make_one_shot_iterator(datasets)
    dataset = iterator.get_next()
    parsed_dataset = tf.io.parse_example(dataset, features={
            'filename': tf.io.FixedLenFeature([], tf.string),
            "x_image": tf.io.FixedLenFeature([], tf.string),
            "y_image": tf.io.FixedLenFeature([], tf.string)})
    x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, batch_size)], tf.float32)
    y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, batch_size)], tf.float32)
    
    # deconv_3 = add_deconv_layer(x_s, [3, 3, 32, 3], [batch_size, input_shape[0] * 2, input_shape[1] * 2, 32], stride=2)

    # 七層CNN 同尺寸 低解 -> 高解
    with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0'):

        # stage 1
        y1 = cnn_with_batch_norm(x_s, [7, 7], 3, 64, 1)

        y2 = cnn_with_batch_norm(y1, [1, 1], 64, 64, 1)

        y3 = cnn_with_batch_norm(y2, [3, 3], 64, 64, 1)

        y4 = cnn_with_batch_norm(y3, [3, 3], 64, 64, 1)

        y5 = cnn_with_batch_norm(y4, [3, 3], 64, 256, 1)

        # stage 1 x
        x_ = cnn_with_batch_norm(x_s, [1,1], 3, 256, 1)

        mean, variance = tf.nn.moments(x_, [0,1,2])
        x_bn = tf.nn.batch_normalization(x_, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-4)
        y1_act = tf.nn.relu(tf.add(y5, x_bn))

        pool = tf.nn.avg_pool(y1_act, ksize=(1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')

    loss = tf.reduce_mean(tf.square(pool - y_s))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(loss)
    # grad_and_var = optimizer.compute_gradients(loss)
    # varis =tf.compat.v1.trainable_variables()
    # gradients = tf.gradients(loss,tf.compat.v1.trainable_variables())
    # gradients = optimizer.compute_gradients(loss)
    # count = 0

    if os.path.isfile("trained_parameters/heigh_resolution_large_to_small_with_fv_" + str(batch_size) + "p.index"):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "trained_parameters/heigh_resolution_large_to_small_with_fv_" + str(batch_size) + "p")
    else:
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
    # dest = '/Users/wen/Desktop/programing/AI_demo/image_generator_train/smaller_100/'
    while True:
        count += 1
        _, loss_val, grad = sess.run([train_step, loss, gradients])
        saver.save(sess=sess, save_path="trained_parameters/heigh_resolution_large_to_small_with_fv_" + str(batch_size) + "p")
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

def train_srcnn_gpu(datasets, batch_size, input_shape, output_shape, channel_size, sess):
    iterator = tf.compat.v1.data.make_one_shot_iterator(datasets)
    dataset = iterator.get_next()
    parsed_dataset = tf.io.parse_example(dataset, features={
            'filename': tf.io.FixedLenFeature([], tf.string),
            "x_image": tf.io.FixedLenFeature([], tf.string),
            "y_image": tf.io.FixedLenFeature([], tf.string)})
    x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, batch_size)], tf.float32)
    y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, batch_size)], tf.float32)
    
    # deconv_3 = add_deconv_layer(x_s, [3, 3, 32, 3], [batch_size, input_shape[0] * 2, input_shape[1] * 2, 32], stride=2)

    # 七層CNN 同尺寸 低解 -> 高解
    # with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0'):

    # stage 1
    y1 = cnn.add_cnn_layer(x_s, [3, 3, 3, 128])

    y2 = cnn.add_cnn_layer(y1, [1, 1, 128, 64])

    y3 = cnn.add_cnn_layer(y2, [3, 3, 64, 3])

    loss = tf.reduce_mean(tf.square(y3 - y_s))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)

    train_step = optimizer.minimize(loss)

    if os.path.isfile("trained_parameters/srcnn_" + str(batch_size) + "p.index"):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess=sess, save_path="trained_parameters/srcnn_" + str(batch_size) + "p")
    else:
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
    while True:
        _, loss_val = sess.run([train_step, loss])
        saver.save(sess=sess, save_path="trained_parameters/srcnn_" + str(batch_size) + "p")
        print(loss_val)

def train_upscale_cnn_gpu(datasets, batch_size, input_shape, output_shape, channel_size, sess):
    iterator = tf.compat.v1.data.make_one_shot_iterator(datasets)
    dataset = iterator.get_next()
    parsed_dataset = tf.io.parse_example(dataset, features={
            'filename': tf.io.FixedLenFeature([], tf.string),
            "x_image": tf.io.FixedLenFeature([], tf.string),
            "y_image": tf.io.FixedLenFeature([], tf.string)})
    x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, batch_size)], tf.float32)
    y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, batch_size)], tf.float32)
    
    # 七層CNN 同尺寸 低解 -> 高解
    # with tf.device('/job:localhost/replica:0/task:0/device:XLA_GPU:0'):

    # stage 1
    y1 = cnn.add_deconv_layer(x_s, [3, 3, 3, 128])

    y2 = cnn.add_cnn_layer(y1, [1, 1, 128, 32])

    y3 = cnn.add_cnn_layer(y2, [3, 3, 32, 3])

    loss = tf.reduce_mean(tf.square(y3 - y_s))
    
    optimizer = tf.compat.v1.train.AdamOptimizer(1e-6)
    train_step = optimizer.minimize(loss)

    if os.path.isfile("trained_parameters/upscale_" + str(batch_size) + "p.index"):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess=sess, save_path="trained_parameters/upscale_" + str(batch_size) + "p")
    else:
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
    while True:
        _, loss_val = sess.run([train_step, loss])
        saver.save(sess=sess, save_path="trained_parameters/upscale_" + str(batch_size) + "p")
        print(loss_val)
