import tensorflow as tf
import numpy as np
import basic_nn_batch as bnn
import os
import cv2
import dataset as dt
import cnn

class HeighResolutionGenerator:
    def __init__(self, datasets, batch_size, input_shape, output_shape, channel_size, model_name='srcnn'):
        self.model_name = model_name
        self.datasets = datasets
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.channel_size = channel_size
        self.g = tf.Graph()
        

    def build_model(self):
        if self.model_name == 'srcnn':
            self.basic_srcnn_model()
        elif self.model_name == 'srcnn_2x':
            self.two_times_srcnn_model()
            
    def basic_srcnn_model(self):
        with self.g.as_default():
            iterator = tf.compat.v1.data.make_one_shot_iterator(self.datasets)
            dataset = iterator.get_next()
            parsed_dataset = tf.io.parse_example(dataset, features={
                    'filename': tf.io.FixedLenFeature([], tf.string),
                    "x_image": tf.io.FixedLenFeature([], tf.string),
                    "y_image": tf.io.FixedLenFeature([], tf.string)})
            x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, self.batch_size)], tf.float32)
            y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, self.batch_size)], tf.float32)

            y1 = cnn.add_cnn_layer(x_s, [3, 3, 3, 128])

            y2 = cnn.add_cnn_layer(y1, [1, 1, 128, 32])

            self.y3 = cnn.add_cnn_layer(y2, [3, 3, 32, 3])

            self.loss = tf.reduce_mean(tf.square(self.y3 - y_s))
                
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
            self.train_step = optimizer.minimize(self.loss)

            self.saver = tf.compat.v1.train.Saver()

            self.sess = tf.Session(graph=self.g)
            if os.path.isfile("trained_parameters/srcnn.index"):
                self.saver.restore(sess=self.sess, save_path="trained_parameters/srcnn")
            else:
                self.sess.run(tf.compat.v1.global_variables_initializer())
    
    def two_times_srcnn_model(self):
        with self.g.as_default():
            iterator = tf.compat.v1.data.make_one_shot_iterator(self.datasets)
            dataset = iterator.get_next()
            parsed_dataset = tf.io.parse_example(dataset, features={
                    'filename': tf.io.FixedLenFeature([], tf.string),
                    "x_image": tf.io.FixedLenFeature([], tf.string),
                    "y_image": tf.io.FixedLenFeature([], tf.string)})
            x_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['x_image'][index]) for index in range(0, self.batch_size)], tf.float32)
            y_s = tf.cast([tf.image.decode_jpeg(parsed_dataset['y_image'][index]) for index in range(0, self.batch_size)], tf.float32)

            y1 = cnn.add_deconv_layer(x_s, [3, 3, 128, 3], [self.batch_size, self.output_shape[0], self.output_shape[1], 128])

            y2 = cnn.add_cnn_layer(y1, [1, 1, 128, 32])

            self.y3 = cnn.add_cnn_layer(y2, [3, 3, 32, 3])

            self.loss = tf.reduce_mean(tf.square(self.y3 - y_s))
                
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
            self.train_step = optimizer.minimize(self.loss)

            self.saver = tf.compat.v1.train.Saver()

            self.sess = tf.Session(graph=self.g)
            if os.path.isfile("trained_parameters/srcnn_2x.index"):
                self.saver.restore(sess=self.sess, save_path="trained_parameters/srcnn_2x")
            else:
                self.sess.run(tf.compat.v1.global_variables_initializer())

    def graph(self):
        return self.g

    def train(self):
        if self.model_name == 'srcnn':
            save_path = "trained_parameters/srcnn"
        elif self.model_name == 'srcnn_2x':
            save_path = "trained_parameters/srcnn_2x"
        try:
            while True:
                _, loss_val = self.sess.run([self.train_step, self.loss])
                self.saver.save(sess=self.sess, save_path=save_path)
                print(loss_val)
        except:
            raise

    def predit(self, image, input_shape, channel_size):

        tf.reset_default_graph
        if self.model_name == 'srcnn':
            return self.srcnn_predit(image, input_shape, channel_size)
        elif self.model_name == 'srcnn_2x':
            return self.two_times_srcnn_predit(image, input_shape, channel_size)
    
    def srcnn_predit(self, image, input_shape, channel_size):
        x_s = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], channel_size], "x_test")

        y1 = cnn.add_cnn_layer(x_s, [3, 3, 3, 128])

        y2 = cnn.add_cnn_layer(y1, [1, 1, 128, 32])

        y3 = cnn.add_cnn_layer(y2, [3, 3, 32, 3])

        pred = tf.cast(y3, tf.uint8)

        saver = tf.compat.v1.train.Saver()

        sess = tf.Session()
        if os.path.isfile("trained_parameters/srcnn.index"):
            saver.restore(sess=sess, save_path="trained_parameters/srcnn")
        else:
            raise '請先完成訓練，或將指定權重放在正確的資料夾下'
        
        out_put = sess.run(pred, feed_dict={x_s: image})

        return out_put[0]
    
    def two_times_srcnn_predit(self):
        x_s = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], channel_size], "x_test")

        y1 = cnn.add_deconv_layer(x_s, [3, 3, 128, 3], [self.batch_size, self.output_shape[0], self.output_shape[1], 128])

        y2 = cnn.add_cnn_layer(y1, [1, 1, 128, 32])

        y3 = cnn.add_cnn_layer(y2, [3, 3, 32, 3])

        pred = tf.cast(y3, tf.uint8)

        saver = tf.compat.v1.train.Saver()

        sess = tf.Session()
        if os.path.isfile("trained_parameters/srcnn_2x.index"):
            saver.restore(sess=sess, save_path="trained_parameters/srcnn_2x")
        else:
            raise '請先完成訓練，或將指定權重放在正確的資料夾下'
        
        out_put = sess.run(pred, feed_dict={x_s: image})

        return out_put[0]
        
        



