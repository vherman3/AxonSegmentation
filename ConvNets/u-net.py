

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import math


IMAGE_SIZE = 80
NUM_CHANNELS = 1
NUM_LABELS = 2
SEED = None

class UNetModel:
    def __init__(self, image_size, depth=4):
        self.depth = depth
        self.num_features = 64
        num_features = self.num_features
        num_features_init = NUM_CHANNELS

        self.num_classes = 2
        self.image_size = image_size

        self.weights_contraction = []
        self.weights_expansion = []
        self.upconv_weights = []

        # Setting variables that will be optimized
        # contraction

        for i in range(self.depth):
            self.weights_contraction.append({'conv1': tf.Variable(tf.random_normal([3, 3, num_features_init, num_features],
                                                                                   stddev=math.sqrt(2.0/(9.0*float(num_features_init))), seed=SEED)),
                                             'bias1': tf.Variable(tf.random_normal([num_features],
                                                                                   stddev=math.sqrt(2.0/(9.0*float(num_features))))),
                                             'conv2': tf.Variable(tf.random_normal([3, 3, num_features, num_features],
                                                                                   stddev=math.sqrt(2.0/(9.0*float(num_features))), seed=SEED)),
                                             'bias2': tf.Variable(tf.random_normal([num_features],
                                                                                   stddev=math.sqrt(2.0/(9.0*float(num_features)))))})
            num_features_init = num_features
            num_features = num_features_init * 2

        self.weights_bottom_layer = {
            'conv1': tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))), seed=SEED)),
            'bias1': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))),
            'conv2': tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))), seed=SEED)),
            'bias2': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))))}

        # expansion
        num_features_init = num_features
        num_features = num_features_init / 2
        for i in range(depth):
            self.upconv_weights.append(tf.Variable(tf.random_normal([2, 2, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))), seed=SEED)))
            self.weights_expansion.append({'conv1': tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))), seed=SEED)),
                                           'bias1': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))),
                                           'conv2': tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))), seed=SEED)),
                                           'bias2': tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))))})
            num_features_init = num_features
            num_features = num_features_init / 2

        self.finalconv_weights = tf.Variable(tf.random_normal([1, 1, num_features * 2, self.num_classes], stddev=math.sqrt(2.0/(9.0*float(num_features*2))), seed=SEED))


    def model(self, data, train=False):
        """The Model definition.
        # 2X 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        # Bias and rectified linear non-linearity.
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        """
        # contraction
        image_size_temp = [self.image_size]
        data_temp = data
        relu_results = []
        for i in range(self.depth):
            conv = tf.nn.conv2d(data_temp, self.weights_contraction[i]['conv1'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_contraction[i]['bias1']))
            conv = tf.nn.conv2d(relu, self.weights_contraction[i]['conv2'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_contraction[i]['bias2']))
            relu_results.append(relu)

            data_temp = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            image_size_temp.append((image_size_temp[-1]) / 2)

        # convolution of bottom layer
        conv = tf.nn.conv2d(data_temp, self.weights_bottom_layer['conv1'], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_bottom_layer['bias1']))
        conv = tf.nn.conv2d(relu, self.weights_bottom_layer['conv2'], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_bottom_layer['bias2']))
        image_size_temp.append(image_size_temp[-1])

        # expansion
        for i in range(self.depth):
            # up-convolution:
            # 2x2 convolution with upsampling by a factor 2, then concatenation
            resample = tf.image.resize_images(relu, image_size_temp[-1] * 2, image_size_temp[-1] * 2)
            upconv = tf.nn.conv2d(resample, self.upconv_weights[i], strides=[1, 1, 1, 1], padding='SAME')
            image_size_temp.append(image_size_temp[-1] * 2)
            upconv_concat = tf.concat(concat_dim=3, values=[tf.slice(relu_results[self.depth-i-1], [0, 0, 0, 0], [-1, image_size_temp[self.depth-i] * 2, image_size_temp[self.depth-i] * 2, -1]), upconv])

            # expansion
            conv = tf.nn.conv2d(upconv_concat, self.weights_expansion[i]['conv1'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_expansion[i]['bias1']))
            conv = tf.nn.conv2d(relu, self.weights_expansion[i]['conv2'], strides=[1, 1, 1, 1], padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.weights_expansion[i]['bias2']))

        finalconv = tf.nn.conv2d(relu, self.finalconv_weights, strides=[1, 1, 1, 1], padding='SAME')
        final_result = tf.reshape(finalconv, tf.TensorShape([finalconv.get_shape().as_list()[0] * image_size_temp[-1] * image_size_temp[-1], NUM_LABELS]))

        return final_result