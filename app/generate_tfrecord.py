import os
import sys
import tensorflow as tf
import glob
import numpy as np
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import dataset

# reader = tf.TFRecordReader()
# filenames = glob.glob('*.tfrecords')
# filename_queue = tf.train.string_input_producer(
#    filenames)
# _, serialized_example = reader.read(filename_queue)
# feature_set = { 'x_image': tf.FixedLenFeature([], tf.string),
#                'y_image': tf.FixedLenFeature([], tf.string)
#            }
           
# features = tf.parse_single_example( serialized_example, features=feature_set )
# x_images = features['x_image']
# y_images = features['y_image']

# print(x_images)

train_x_file_path = os.path.join(os.getcwd(),'image_generator_train/low_resol_720p')
train_y_file_path = os.path.join(os.getcwd(),'image_generator_train/height_360p')

dataset.img_to_small_size_tfrecord(train_x_file_path, train_y_file_path)

# tf.enable_eager_execution()
# filenames = glob.glob('*.tfrecords')
# datasets = tf.data.TFRecordDataset(filenames)

# # print(datasets.batch(10))
# for dataset in datasets.batch(10):
#    parsed = tf.parse_example(dataset, features={
#    "x_image": tf.FixedLenFeature((), tf.string, default_value=""),
#    "y_image": tf.FixedLenFeature((), tf.string, default_value="")
#    })

#    x_imgs = tf.decode_raw(parsed['x_image'], tf.uint8)

#    print()