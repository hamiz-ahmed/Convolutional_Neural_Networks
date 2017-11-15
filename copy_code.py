from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def compute_accuracy(sess, validation_xs, validation_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: validation_xs, keep_prob:1})
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


def perform_operations():
    learning_rates = [0.1, 0.01, 0.001, 0.0001]

    for rate in learning_rates:
        # define session
        sess = tf.Session()

        # initialize global tf variables
        init = tf.global_variables_initializer()
        sess.run(init)

        print("Learning rate: ", rate)
        y_plot_validation = []
        x_plot_validation = []

        for i in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, learning_rate: 0.1, keep_prob: 0.5})
            if i % 50 == 0:
                print("Step: ", i)
                accuracy = compute_accuracy(sess, mnist.validation.images, mnist.validation.labels)
                print("Validation accuracy: ", accuracy)
                x_plot_validation.append(i)
                y_plot_validation.append(accuracy)

        plt.plot(x_plot_validation, y_plot_validation, label=rate)
        sess.close()

    # for plotting graph
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # Load mnist data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # define placeholder for inputs to network
    learning_rate = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 784]) / 255.  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(xs, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)

    # conv layer 1
    W_conv1 = weight_variable([3, 3, 1, 16])  # patch 3x3, in size 1, out size 16
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x16
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x16

    # conv layer 2
    W_conv2 = weight_variable([3, 3, 16, 32])  # patch 3x3, in size 16, out size 32
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x32
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x32

    # fully connected layer
    W_fc1 = weight_variable([7 * 7 * 32, 128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
    h_fc1 = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Softmax output layer
    W_fc2 = weight_variable([128, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
