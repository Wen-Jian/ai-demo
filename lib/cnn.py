import tensorflow as tf

def add_cnn_layer(xs, convolute_shape):
    cnn_filter = tf.Variable(tf.truncated_normal(convolute_shape, stddev=0.1))
    cnn_bias = tf.Variable(tf.random_normal(convolute_shape))
    before_pooling = tf.nn.conv2d(xs, cnn_filter, strides=[1,1,1,1], padding='SAME') 
    pooling = tf.nn.max_pool(before_pooling, ksize=[1,2,2,1],strides=[1,2,2,1])
    tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)

