import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from pylab import savefig
import random

from input_data import input_data
data_train = input_data('train')
data_test = input_data('test')


result_number = 5
folder = 'dataset/image_results_%s'%result_number
if not os.path.exists(folder):
    os.makedirs(folder)

folder2 = 'dataset/model_parameters%s'%result_number
if not os.path.exists(folder2):
    os.makedirs(folder2)

# Divers variables
Loss = []
Step = []
text = ''

# Parameters
learning_rate = 0.003
training_iters = 1000
batch_size = 1
display_step = 50
save_step = 50
depth = 4
image_size = 256
number_of_cores = 3

# Network Parameters
n_input = image_size * image_size
n_classes = 2
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
y = tf.placeholder(tf.float32, shape=(batch_size*n_input, n_classes))
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout, image_size = image_size):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, image_size, image_size, 1])
    data_temp = x
    data_temp_size = [image_size]
    relu_results = []

# contraction
    for i in range(depth):
      conv1 = conv2d(data_temp, weights['wc1'][i], biases['bc1'][i])
      conv2 = conv2d(conv1, weights['wc2'][i], biases['bc2'][i])
      relu_results.append(conv2)

      conv2 = maxpool2d(conv2, k=2)
      data_temp_size.append(data_temp_size[-1]/2)
      data_temp = conv2

    conv1 = conv2d(data_temp, weights['wb1'], biases['bb1'])
    conv2 = conv2d(conv1, weights['wb2'], biases['bb2'])
    data_temp_size.append(data_temp_size[-1])
    data_temp = conv2


# expansion
    for i in range(depth):
        data_temp = tf.image.resize_images(data_temp, data_temp_size[-1] * 2, data_temp_size[-1] * 2)
        upconv = conv2d(data_temp, weights['upconv'][i], biases['upconv'][i])
        #upconv = tf.nn.conv2d(data_temp, weights['upconv'][i], strides=[1, 1, 1, 1], padding='SAME')
        data_temp_size.append(data_temp_size[-1]*2)

        upconv_concat = tf.concat(concat_dim=3, values=[tf.slice(relu_results[depth-i-1], [0, 0, 0, 0],
                                                                 [-1, data_temp_size[depth-i-1], data_temp_size[depth-i-1], -1]), upconv])
        conv1 = conv2d(upconv_concat, weights['we1'][i], biases['be1'][i])
        conv2 = conv2d(conv1, weights['we2'][i], biases['be2'][i])
        data_temp = conv2

    #finalconv = conv2d(conv2, weights['finalconv'], biases['finalconv_b'])
    finalconv = tf.nn.conv2d(conv2, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
    final_result = tf.reshape(finalconv, tf.TensorShape([finalconv.get_shape().as_list()[0] * data_temp_size[-1] * data_temp_size[-1], 2]))

    return final_result


weights = {'wc1':[],'wc2':[],'we1':[],'we2':[],'upconv':[],'finalconv':[],'wb1':[], 'wb2':[]}
biases = {'bc1':[],'bc2':[],'be1':[],'be2':[],'finalconv_b':[],'bb1':[], 'bb2':[],'upconv':[]}


# Contraction
for i in range(depth):
  if i == 0:
    num_features_init = 1
    num_features = 64
  else:
    num_features = num_features_init * 2


# Store layers weight & bias

  weights['wc1'].append(tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))), name = 'wc1-%s'%i))
  weights['wc2'].append(tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name = 'wc2-%s'%i))
  biases['bc1'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='bc1-%s'%i))
  biases['bc2'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='bc2-%s'%i))

  image_size = image_size/2
  num_features_init = num_features
  num_features = num_features_init*2

weights['wb1']= tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))),name='wb1-%s'%i)
weights['wb2']= tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='wb2-%s'%i)
biases['bb1']= tf.Variable(tf.random_normal([num_features]), name='bb2-%s'%i)
biases['bb2']= tf.Variable(tf.random_normal([num_features]), name='bb2-%s'%i)

num_features_init = num_features

for i in range(depth):

    num_features = num_features_init/2
    weights['upconv'].append(tf.Variable(tf.random_normal([2, 2, num_features_init, num_features]), name='upconv-%s'%i))
    biases['upconv'].append(tf.Variable(tf.random_normal([num_features]), name='bupconv-%s'%i))
    weights['we1'].append(tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))), name='we1-%s'%i))
    weights['we2'].append(tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='we2-%s'%i))
    biases['be1'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='be1-%s'%i))
    biases['be2'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))), name='be2-%s'%i))

    num_features_init = num_features

weights['finalconv']= tf.Variable(tf.random_normal([1, 1, num_features, n_classes]), name='finalconv-%s'%i)
biases['finalconv_b']= tf.Variable(tf.random_normal([n_classes]), name='bfinalconv-%s'%i)

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
tf.scalar_summary('Loss', cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())


summary_op = tf.merge_all_summaries()

# Launch the graph
with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads= number_of_cores, intra_op_parallelism_threads= number_of_cores)) as sess:
    sess.run(init)
    step = 1

    print 'training start'

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = data_train.next_batch(batch_size, rnd = True)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc, p = sess.run([cost, accuracy, pred], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})

            prediction = data_train.read_batch(p, batch_size)[0, :, :, 0]
            ground_truth = data_train.read_batch(batch_y, batch_size)[0, :, :, 0]

            Loss.append(loss)
            Step.append(step)

            if step % save_step == 0 :
                image = batch_x[0, :, :]
                plt.figure(1)
                plt.imshow(image, cmap=plt.get_cmap('gray'))
                plt.hold(True)
                plt.imshow(prediction, alpha=0.7)
                savefig(folder+'/prediction_%s'%step+'.png')

                image = batch_x[0, :, :]
                plt.figure(2)
                plt.imshow(image, cmap=plt.get_cmap('gray'))
                plt.hold(True)
                plt.imshow(ground_truth, alpha=0.7)
                savefig(folder+'/GT_%s'%step+'.png')

                #if step == save_step :
                    #plt.show()


            if step%(3*save_step) == 0 :
                save_path = saver.save(sess, folder2+"/model.ckpt")
                print("Model saved in file: %s" % save_path)
                #plt.show()



            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)

        step += 1
    print "Optimization Finished!"


    print Step
    print Loss
    plt.plot(Step[3:], Loss[3:])
    savefig(folder+'/loss_evolution'+'.png')

    # Calculate accuracy for 256 mnist test images
    test_accuracy = []
    for i in range(100):
        batch_x,batch_y = data_test.next_batch(batch_size, rnd = True)
        test_accuracy.append(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.}))
    print 'mean accuracy', np.mean(test_accuracy)

    f = open(folder+'/report'+'.txt', 'w')
    f.write(text)
    f.close()


