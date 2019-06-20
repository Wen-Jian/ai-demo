import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import glob
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import create_mnist_jpg as im_creator
import basic_nn_batch as batch_nn
import cnn 
import cnn_gpu
from PIL import Image
import dataset as dt
import cv2
from skimage import exposure

batch_size = 1
parameter_name = 'heigh_resolution_large_to_small_100p'
# # 訓練代碼 用tfrecord
filenames = glob.glob('img_small_data_2x.tfrecords')
datasets = tf.data.TFRecordDataset(filenames).repeat(100).shuffle(1000).batch(batch_size)

sess = tf.Session()
cnn_gpu.train_heigh_resolution_with_gpu(datasets, batch_size, (360, 640), (360, 640), 3, sess)

# 預訓練影像生成
# filenames = glob.glob('img_small_data_2x.tfrecords')
# datasets = tf.data.TFRecordDataset(filenames).repeat(300).shuffle(200).batch(batch_size)

# sess = tf.Session()
# out_put = cnn.img_generator_train_with_imgs(datasets, batch_size, (360, 640), (720, 1280), 3, sess)



# 用原始影像訓練
# x_images = glob.glob('image_generator_train/x_small/*.jpg')
# x_images.sort()
# y_images = glob.glob('image_generator_train/y/*.jpg')
# y_images.sort()
# x_img_data_array = []
# y_img_data_array = []

# for index in range(0, 10):
#     print("round %d" % index)
#     for i in range(0, len(x_images)):
#         x_data = im_creator._parse_function(x_images[i])/255.
#         x_shape = np.shape(x_data)
#         x_img_data_array.append(np.reshape(x_data, [x_shape[0] * x_shape[1] * x_shape[2]]))
#         y_data = im_creator._parse_function(y_images[i])
#         y_shape = np.shape(y_data)
#         y_img_data_array.append(np.reshape(y_data, [y_shape[0] * y_shape[1] * y_shape[2]]))
#         percentage = (i/batch_size)*100
#         if i > 0 and i % (batch_size -1) == 0:
#             cnn.heigh_resolution_generator_train_with_imgs(x_img_data_array, y_img_data_array, batch_size, (108, 192), (1080, 1920), 3)
#             x_img_data_array = []
#             y_img_data_array = []



# 生成測試影像
# test_file_path = glob.glob('image_generator_train/x_360p/0032.jpg')
# x_img = cv2.imread(test_file_path[0])
# sess = tf.Session()
# predit = cnn.generate_image([x_img], 1, (360, 640), (720, 1280), 3, sess, parameter_name)

# # (bB, bG, bR) = cv2.split(x_img)

# # mbR = bR.mean()
# # mbG = bG.mean()
# # mbB = bB.mean()

# (B, G, R) = cv2.split(predit)
# # mR = R.mean()
# # print(mR)
# # mG = G.mean()
# # print(mG)
# # mB = B.mean()
# # print(mB)

# # KB = (mR + mG + mB) / (3 * mB);
# # KG = (mR + mG + mB) / (3 * mG);
# # KR = (mR + mG + mB) / (3 * mR);
# # print(R)
# # R = (R * 1.)
# # R[::] = 255
# # print(R)
# # print(G)
# # G = (G * 1.5)
# # G[::] = 255
# # print(G)
# # print(B)
# # B = (B * 1.)
# # B[::] = 255
# # print(B)

# # adjusted_predit = exposure.rescale_intensity((cv2.merge([B, G, R])).astype('uint8'))
# # adjusted_predit = cv2.merge([B, G, R]).astype('uint8')
# cv2.imshow('before', (predit).astype('uint8'))
# cv2.imshow('adjusted_predit', (cv2.merge([B, G, R]).astype('uint8') * 1.).astype('uint8'))
# cv2.imshow('origin', x_img)
# # # cv2.imwrite('image_generator_train/output.jpg', predit)
# # print(adjusted_predit[::1])
    
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# sess.close()

# vidcap = cv2.VideoCapture('video/【娛樂百分百】2019_05_10《凹嗚狼人殺》愷樂、宇辰│梁以辰、黃少谷、大鶴、紀卜心、嘻小瓜、楊上昀、邱鋒澤、孫生 (360p).mp4')
# parameter_name = 'heigh_resolution_1_deconv_2x_500p'
# count = 0
# success = True
# sess = tf.Session()
# while success:
# #     success,image = vidcap.read()
# #     x_img = [image]
# #     shape = np.shape(image)
# #     sess = tf.Session()
# #     shape = np.shape(cv2.imread(test_file_path[0]))
# #     sess = tf.Session()
# #     print((shape[0] * 2, shape[1] * 2))
# #     predit = cnn.generate_image(x_img, 1, (108, 192), (216, 384), 3, sess, parameter_name)
#         test_file_path = glob.glob('image_generator_train/x_small/0020.jpg')
#         x_img = [cv2.imread(test_file_path[0])]
#         predit = cnn.generate_image(x_img, 1, (108, 192), (216, 384), 3, sess, parameter_name)
#         print(predit)
#         # cv2.imshow('image_test', cv2.imread(test_file_path[0]))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


# # eager_execution mode for training
# for i in range(0, 1000):
# #     datasets = tf.data.TFRecordDataset(filenames).batch(batch_size)
#     for dataset in datasets:
# #         input_shape = (192, 108)
# #         output_shape = (1920, 1080)
# #         channel_size = 3
#         parsed_dataset = tf.parse_example(dataset, features={
#             'filename': tf.FixedLenFeature([], tf.string),
#             "x_image": tf.FixedLenFeature([], tf.string),
#             "y_image": tf.FixedLenFeature([], tf.string)})
#         filenames = [file_name for file_name in parsed_dataset['filename']]
#         x_image_decoded = [tf.image.decode_image(image_string) for image_string in parsed_dataset['x_image']]
#         y_image_decoded = [tf.image.decode_image(image_string) for image_string in parsed_dataset['y_image']]
#         for index in range(0, len(y_image_decoded)):
#                 print(filenames[index])
#                 print(np.shape(x_image_decoded[index]))
                
#     break
#         cv2.imshow('test', np.array(x_image_decoded[0]))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         x_imgs = tf.decode_raw(parsed_dataset['x_image'], tf.uint8)
#         x_imgs = np.reshape(x_imgs, (np.shape(x_imgs)[0], input_shape[0], input_shape[1], channel_size))
#         y_imgs = tf.decode_raw(parsed_dataset['y_image'], tf.float32)
#         print(np.shape(y_imgs))
#     break
#         y_imgs = np.reshape(y_imgs, (np.shape(y_imgs)[0], output_shape[0], output_shape[1], channel_size))
#         sess = tf.Session()
#         cnn.train_heigh_resolution_generator(x_imgs, y_imgs, input_shape, output_shape, channel_size, sess)
