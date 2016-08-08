import os
import matplotlib.pyplot as plt
import tensorflow as tf
from input_data import input_data

data_test = input_data('test')
result_number = 5

batch_size = 1
display_step = 50
depth = 4
image_size = 256

dropout = 0.75 # Dropout, probability to keep units

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "dataset/model_parameters5/model.ckpt")


