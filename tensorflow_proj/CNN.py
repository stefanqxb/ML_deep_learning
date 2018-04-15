
# note: when it comes multiple layers of nernural network, you need
# to be careful choosing different AF, because it will cause problem
# like gradient explosive or gradient vanish

# some advices: CNN perfer relu and RNN perfer tanh or relu

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape, name="W"):
    with tf.name_scope(name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="weight")


def bias_variable(shape, name="B"):
    with tf.name_scope(name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="bias")


def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def add_conv_layer(input, mask_size, input_size, output_size, activation_func, name="conv"):
    with tf.name_scope(name):
        weights = weight_variable([mask_size, mask_size, input_size, output_size], name="W")
        bias = bias_variable([output_size], name="B")
        h_conv = activation_func(conv2d(input, weights) + bias)
        h_pool = max_pool_2x2(h_conv)
        tf.summary.histogram('weight', weights)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('activations', h_conv)
        return h_pool


def add_dense_layer(input, input_size, output_size, activation_func, name="fc1"):
    with tf.name_scope(name):
        weights = weight_variable([input_size, output_size], name="W")
        bias = bias_variable([output_size], name="B")
        h_pool2_flat = tf.reshape(input, [-1, input_size])
        h_fc1 = activation_func(tf.matmul(h_pool2_flat, weights) + bias)
        return h_fc1


def add_output_layer(input, input_size, output_size, name="fc"):
    with tf.name_scope(name):
        weights = weight_variable([input_size, output_size], name="W")
        bias = bias_variable([output_size], name="B")
        y_conv = tf.matmul(input, weights) + bias
        return y_conv


def setup_cnn_training_model(x_image, keep_prob):
    """ set up convolutional layer and pooling layer """
    #  Convolutional layer

    h_pool1 = add_conv_layer(x_image, 5, 1, 32, tf.nn.relu, name="conv1")

    h_pool2 = add_conv_layer(h_pool1, 5, 32, 64, tf.nn.relu, name="conv2")

    # densely(fully)-connected layer

    h_fc1 = add_dense_layer(h_pool2, 7 * 7 * 64, 1024, tf.nn.relu, name="fc1")

    # dropout to reduce over-fitting

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    y_conv = add_output_layer(h_fc1_drop, 1024, 10, "fc2")
    return y_conv


def evaluate_training_result(expected_output, trained_result):
    """ set up evaluation of the training result """
    with tf.name_scope("entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=expected_output, logits=trained_result))
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope("accuray"):
        correct_prediction = tf.equal(tf.argmax(trained_result, 1), tf.argmax(expected_output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return train_step, accuracy


def main():
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="label")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32)
    tarining_times = 500

    y_conv = setup_cnn_training_model(x_image, keep_prob)

    train_step, accuracy = evaluate_training_result(y_, y_conv)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merge_summary = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter('/home/exuaqiu/xuanbin/ML/tensorflow_proj/graph1')
        for i in range(tarining_times):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                s = sess.run(merge_summary, feed_dict={x: batch[0], y_: batch[1]})
                file_writer.add_summary(s, i)
                accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        tf.summary.image('input', x_image, 3)


if __name__ == '__main__':
    main()

