import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
import pickle
from pylab import savefig
from im2batch import im2batch, batch2im
import numpy as np
import scipy
import time
from mrf import train_mrf, run_mrf
from sklearn.metrics import accuracy_score
from segmentation_scoring import rejectOne_score

from input_data import input_data
path_img = '/Users/viherm/Desktop/small_one.png'
data_test = input_data('test')

train = True

batch_size = 1
depth = 6
image_size = 256
number_of_cores = 3
n_input = image_size * image_size
n_classes = 2
model_number = 13


folder_model = 'dataset/model_parameters%s'%model_number
if not os.path.exists(folder_model):
    os.makedirs(folder_model)

x = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size))
y = tf.placeholder(tf.float32, shape=(batch_size*n_input, n_classes))
keep_prob = tf.placeholder(tf.float32)


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

saver = tf.train.Saver(tf.all_variables())

# Image to batch
image_init, data, positions = im2batch(path_img, 256, rescale_coeff=1.0)
print len(data)
predictions = []

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, folder_model+"/model.ckpt")

    for i in range(len(data)):
        print 'iteration %s on %s'%(i, len(data))
        batch_x = np.asarray([data[i]])

        start = time.time()
        p = sess.run(pred, feed_dict={x: batch_x})
        print time.time() - start,'s - test time'
        prediction_m = p[:, 0].reshape(256,256)
        prediction = p[:, 1].reshape(256,256)

        Mask = prediction - prediction_m > 0
        predictions.append(Mask)

        image = batch_x[0, :, :]
        plt.figure(2)
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(Mask, alpha=0.7)

        image = batch_x[0, :, :]
        plt.figure(3)
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.hold(True)
        plt.imshow(prediction-prediction_m, alpha=0.7)
        #plt.show()

h_size, w_size = image_init.shape
prediction = batch2im(predictions, positions, h_size, w_size)

plt.figure(3)
plt.imshow(image_init, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(prediction, alpha=0.7)



#######################################################################################################################
#                                            Mrf                                                        #
#######################################################################################################################


folder = os.path.abspath(os.path.join(os.pardir))
data = pickle.load(open(folder +"/data/groundTruth.pkl", "rb"))
mask = data['mask']

nb_class = 2
max_map_iter = 10
alpha = 1.0
beta = 1.0
sigma_blur = 1.0
threshold_learning = 0.1
threshold_sensitivity = 0.65
threshold_error = 0.10
y_pred = prediction.reshape(-1, 1)

folder_mrf = 'dataset/mrf_parameters'
if not os.path.exists(folder_mrf):
    os.makedirs(folder_mrf)

if train :
    y_pred_train = y_pred.copy()
    y_train = mask.reshape(-1,1)
    weight = train_mrf(y_pred_train,image_init , nb_class, max_map_iter, [alpha, beta, sigma_blur], threshold_learning, y_train, threshold_sensitivity)
    mrf_coef = {'weight':weight}

    with open(folder_mrf+'/mrf_parameter.pkl', 'wb') as handle:
         pickle.dump(mrf_coef, handle)

else :
    weight = pickle.load(open(folder_mrf +'/mrf_parameter.pkl', "rb"))['weight']


img_mrf = run_mrf(y_pred, image_init, nb_class, max_map_iter, weight)
img_mrf = img_mrf == 1

if train :
    print 'accuracy before post-processing :',accuracy_score(y_pred, mask.reshape(-1,1))
    print 'accuracy :',accuracy_score(img_mrf.reshape(-1,1), mask.reshape(-1,1))
    print 'reject one :', rejectOne_score(image_init, mask.reshape(-1,1), img_mrf.reshape(-1,1), visualization=False, min_area=10, show_diffusion = True)

plt.figure(4)
plt.imshow(image_init, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(img_mrf, alpha=0.7)
plt.savefig('out_mrf.png', bbox_inches='tight', pad_inches=0)


plt.figure(5)
plt.imshow(image_init, cmap=plt.get_cmap('gray'))
plt.hold(True)
plt.imshow(prediction, alpha=0.7)
plt.savefig('out.png', bbox_inches='tight', pad_inches=0)

plt.show()









