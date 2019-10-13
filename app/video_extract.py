import cv2
import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'lib'))
import cnn
import glob

vidcap = cv2.VideoCapture('video/【娛樂百分百】2019_05_10《凹嗚狼人殺》愷樂、宇辰│梁以辰、黃少谷、大鶴、紀卜心、嘻小瓜、楊上昀、邱鋒澤、孫生 (360p).mp4')
# parameter_name = 'heigh_resolution_large_to_small_100p'
parameter_name = 'upscale_128_3_layer_10p'
count = 0
success = True
while True:
    success,image = vidcap.read()
    x_img = [image]
    shape = np.shape(image)
    sess = tf.Session()
    predit = cnn.generate_image(x_img, 1, shape, (shape[0] * 2, shape[1] * 2), 3, sess, parameter_name)
    cv2.imshow('origin', image)
    cv2.imshow('image_test', predit)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    # tf.reset_default_graph()

    # cv2.imwrite("video/video_y/frame%05d.jpg" % count, image)     # save frame as JPEG file
    # success,image = vidcap.read()
    # print('Read a new frame: ', success)
    # count += 1

# When everything done, release the capture
vidcap.release()
cv2.destroyAllWindows()