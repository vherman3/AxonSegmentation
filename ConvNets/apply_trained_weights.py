import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from u_net import conv_net

from input_data import input_data

data_test = input_data('test')
result_number = 5

batch_size = 1
display_step = 50
depth = 4
image_size = 256

# Network Parameters
n_input = image_size * image_size
n_classes = 2
dropout = 0.75 # Dropout, probability to keep units

x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
y = tf.placeholder(tf.float32, shape=(batch_size*n_input, n_classes))
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

weights = {'wc1':[],'wc2':[],'we1':[],'we2':[],'upconv':[],'finalconv':[],'wb1':[], 'wb2':[]}
biases = {'bc1':[],'bc2':[],'be1':[],'be2':[],'finalconv_b':[],'bb1':[], 'bb2':[],'upconv':[]}

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "dataset/model_parameters5/model.ckpt")
    #a = tf.get_variable("wc1-0")
    #print a

    for x in tf.all_variables():
        print x.name
    for i in range(depth):
      #weights['wc1'].append(tf.Variable(name = ))
      weights['wc2'].append(tf.Variable())
      biases['bc1'].append(tf.Variable())
      biases['bc2'].append(tf.Variable())

    weights['wb1']= tf.Variable()
    weights['wb2']= tf.Variable()
    biases['bb1']= tf.Variable()
    biases['bb2']= tf.Variable()


for i in range(depth):
    weights['upconv'].append(tf.Variable(name = 'upconv-%s'%i))
    biases['upconv'].append(tf.Variable())
    weights['we1'].append(tf.Variable())
    weights['we2'].append(tf.Variable())
    biases['be1'].append(tf.Variable())
    biases['be2'].append(tf.Variable())


weights['finalconv']= tf.Variable()
biases['finalconv_b']= tf.Variable()

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    saver.restore(sess, "dataset/model_parameters1")
    sess.run(accuracy, feed_dict={x: data_test.extract_batch(0, batch_size)[0],
                                      y: data_test.extract_batch(0, batch_size)[1],
                                      keep_prob: 1.})