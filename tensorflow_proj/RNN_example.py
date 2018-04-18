import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
tf.set_random_seed(1)


lr = 0.001
training_iters = 100000
# number of input image each time
batch_size = 128
# image size 28 row *28 column
# each time, send in one row with all column, repeat (row) times
n_inputs = 28 # 28 column
n_steps = 28 # 28 row
n_hidden_units = 128
# digits 0-9
n_classes = 10


def init_data():
    """ set up init data for RNN model """
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_inputs])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
    weights = {'in': tf.Variable(tf.random_normal(shape=[n_inputs, n_hidden_units])),
               'out': tf.Variable(tf.random_normal(shape=[n_hidden_units, n_classes]))
               }
    biases = {'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
              'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
              }

    return x, y, weights, biases


def RNN(X, weights, biases):
    """ convert original data to a different shape """
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in'] #  W*X + b
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # Use basic RNN LSTM cell
    # lstm cell is divided into two parts (c_state, h_state) (zhuxian zhuangtai, fenxian zhuangtai)
    _init_state, lstm_cell = generate_lstm_cell()

    # time_major has different value depends on the input format:
    #  False for inputs =(batches, steps, inputs), True for (steps, batches, inputs)
    output, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    results = tf.matmul(final_state[1], weights['out']) + biases['out'] # final_state[1] fen xian ju qing final state

    return results


def generate_lstm_cell():
    """ generate cell for RNN model """
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    return _init_state, lstm_cell


def main():

    # inital status
    x, y, weights, biases = init_data()
    pred = RNN(x, weights, biases)

    # define cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    # set up trainer
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step*batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys, })
            if step % 20 == 0:
                print("The accuracy in step " + str(step) + " is ")
                print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, }))
            step += 1


if __name__ == '__main__':
    main()