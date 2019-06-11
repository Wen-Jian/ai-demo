import tensorflow as tf
import os
import glob
import sys
from PIL import Image
import re

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def img_to_tfrecord(x_files_path, y_files_path):
    tfrecord_filename = 'img_data.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    x_images = glob.glob(os.path.join(x_files_path, '*.jpg'))
    y_images = glob.glob(os.path.join(y_files_path, '*.jpg'))
    for index in range(0, len(x_images)):
        x_img = Image.open(x_images[index])
        y_img = Image.open(y_images[index])
        feature = { 'x_image': _bytes_feature(x_img.tobytes()),
                    'y_image': _bytes_feature(y_img.tobytes()) }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
def img_to_small_size_tfrecord(x_files_path, y_files_path):
    tfrecord_filename = 'img_small_data_2x.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    x_images = glob.glob(os.path.join(x_files_path, '*.jpg'))
    x_images.sort()
    y_images = glob.glob(os.path.join(y_files_path, '*.jpg'))
    y_images.sort()
    for index in range(0, len(x_images)):
        with tf.gfile.FastGFile(x_images[index], 'rb') as fid:
            x_img = fid.read()
        with tf.gfile.FastGFile(y_images[index], 'rb') as fid:
            y_img = fid.read()
        if re.search('\d*\.jpg', x_images[index]).group() != re.search('\d*\.jpg', y_images[index]).group():
            print(x_images[index])
            print(y_images[index])
            break
        feature = { 
            'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [x_images[index].encode('utf-8')])),
            'x_image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [x_img])),      
            'y_image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [y_img])) }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    
def tfrecord_batch_to_image(parsed_dataset):
    x_image_decoded = [tf.image.decode_image(image_string) for image_string in parsed_dataset['x_image']]
    y_image_decoded = [tf.image.decode_image(image_string) for image_string in parsed_dataset['y_image']]
    return { "x_image": x_image_decoded, "y_image": y_image_decoded} 

