from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time


def compute_accuracy(sess, validation_xs, validation_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: validation_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(validation_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: validation_xs, ys: validation_ys, keep_prob:1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def perform_sgd(sess):
    for i in range(5000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
        if i % 50 == 0:
            print("Step: ", i)
            accuracy = compute_accuracy(sess, mnist.validation.images, mnist.validation.labels)
            print("Validation accuracy: ", accuracy)


def show_graph(x, y):
    plt.plot(x, y)
    plt.xlabel('parameters')
    plt.ylabel('runtime')
    plt.show()


if __name__ == '__main__':
    # Load mnist data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 784]) / 255.  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    num_filters = [8, 16, 32, 64]
    runtimes = []
    parameters = []

    for filter_size in num_filters:
        print("Filter size: ", filter_size)

        #start logging time
        start_time = time.time()

        # conv layer 1 #160
        W_conv1 = weight_variable([3, 3, 1, filter_size])  # patch 3x3, in size 1, out size 16, 16 filters initially
        b_conv1 = bias_variable([filter_size])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x16
        h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x16
        parameters_1 = (3*3*filter_size) + filter_size

        # conv layer 2 320
        W_conv2 = weight_variable([3, 3, filter_size, (filter_size*2)])  # patch 3x3, in size 16, out size 32
        b_conv2 = bias_variable([filter_size*2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x32
        h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x32
        parameters_2 = parameters_1*2

        # fully connected layer 200832
        W_fc1 = weight_variable([7 * 7 * (filter_size*2), 128])
        b_fc1 = bias_variable([128])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * (filter_size*2)])
        h_fc1 = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        parameters_3 = (7*7*(filter_size*2)*128) + 128

        # Softmax output layer
        W_fc2 = weight_variable([128, 10])
        b_fc2 = bias_variable([10])
        prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        parameters_4 = (128*10)+10

        total_parameters = parameters_1 + parameters_2 + parameters_3 + parameters_4
        parameters.append(total_parameters)

        # the error between prediction and real data
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                      reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

        sess = tf.Session()
        # initialize global tf variables
        init = tf.global_variables_initializer()
        sess.run(init)

        perform_sgd(sess)

        sess.close()
        runtimes.append((time.time()-start_time)/60)

show_graph(parameters, runtimes)




