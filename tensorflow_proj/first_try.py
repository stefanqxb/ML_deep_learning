#!/usr/bin/python
from __future__ import print_function
import tensorflow as tf
import numpy as np


def create_data():
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.1 + 0.3
    return x_data, y_data


def create_tf_structure():
    Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    biases = tf.Variable(tf.zeros([1]))
    return Weights, biases


def main():
    x_data, y_data = create_data()
    weights, bias = create_tf_structure()

    y = weights*x_data + bias
    loss = tf.reduce_mean(tf.square(y-y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    session = tf.Session()
    session.run(init) # activate the structure

    for step in range(200):
        session.run(train)
        if step % 20 == 0:
            print(step, session.run(weights), session.run(bias))


if __name__ == '__main__':
    main()
