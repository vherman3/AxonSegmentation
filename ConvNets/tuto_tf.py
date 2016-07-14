import tensorflow as tf
import math
import matplotlib.pyplot as plt

from input_data import input_data
data_train = input_data('train')
data_test = input_data('test')


# Parameters
learning_rate = 0.05
training_iters = 1000
batch_size = 5
display_step = 1
depth = 3
image_size = 256

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
        upconv = tf.nn.conv2d(data_temp, weights['upconv'][i], strides=[1, 1, 1, 1], padding='SAME')
        data_temp_size.append(data_temp_size[-1]*2)

        #print upconv
        #print relu_results[depth-i-1]
        #print tf.slice(relu_results[depth-i-1], [0, 0, 0, 0], [-1, data_temp_size[depth-i-1], data_temp_size[depth-i-1], -1])

        upconv_concat = tf.concat(concat_dim=3, values=[tf.slice(relu_results[depth-i-1], [0, 0, 0, 0],
                                                                 [-1, data_temp_size[depth-i-1], data_temp_size[depth-i-1], -1]), upconv])
        conv1 = conv2d(upconv_concat, weights['we1'][i], biases['be1'][i])
        conv2 = conv2d(conv1, weights['we2'][i], biases['be2'][i])
        data_temp = conv2

    finalconv = tf.nn.conv2d(conv2, weights['finalconv'], strides=[1, 1, 1, 1], padding='SAME')
    final_result = tf.reshape(finalconv, tf.TensorShape([finalconv.get_shape().as_list()[0] * data_temp_size[-1] * data_temp_size[-1], 2]))

    return final_result


weights = {'wc1':[],'wc2':[],'we1':[],'we2':[],'upconv':[],'finalconv':[],'wb1':[], 'wb2':[]}
biases = {'bc1':[],'bc2':[],'be1':[],'be2':[],'bb1':[], 'bb2':[]}


# Contraction
for i in range(depth):
  if i == 0:
    num_features_init = 1
    num_features = 64
  else:
    num_features = num_features_init * 2


# Store layers weight & bias

  weights['wc1'].append(tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))))))
  weights['wc2'].append(tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))))
  biases['bc1'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))))
  biases['bc2'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))))

  image_size = image_size/2
  num_features_init = num_features
  num_features = num_features_init*2

weights['wb1']= tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init)))))
weights['wb2']= tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features)))))
biases['bb1']= tf.Variable(tf.random_normal([num_features]))
biases['bb2']= tf.Variable(tf.random_normal([num_features]))

num_features_init = num_features

for i in range(depth):

    num_features = num_features_init/2
    weights['upconv'].append(tf.Variable(tf.random_normal([2, 2, num_features_init, num_features])))
    weights['we1'].append(tf.Variable(tf.random_normal([3, 3, num_features_init, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features_init))))))
    weights['we2'].append(tf.Variable(tf.random_normal([3, 3, num_features, num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))))
    biases['be1'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))))
    biases['be2'].append(tf.Variable(tf.random_normal([num_features], stddev=math.sqrt(2.0/(9.0*float(num_features))))))

    num_features_init = num_features

weights['finalconv']= tf.Variable(tf.random_normal([1, 1, num_features, n_classes]))


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = data_train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc, p = sess.run([cost, accuracy, pred], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})

            prediction = data_train.read_batch(p, batch_size)[0, :, :, 1]
            image = batch_x[0, :, :]
            plt.figure(5)
            plt.imshow(image, cmap=plt.get_cmap('gray'))
            plt.hold(True)
            plt.imshow(prediction, alpha=0.7)
            plt.show()



            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)

        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 256 mnist test images
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: data_test.extract_batch(0, batch_size)[0],
                                      y: data_test.extract_batch(0, batch_size)[1],
                                      keep_prob: 1.})
