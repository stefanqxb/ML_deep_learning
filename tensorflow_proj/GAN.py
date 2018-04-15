from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 100000
batch_size = 4
learning_rate = 0.0001

# Network Params
image_dim = 250000 # 500*500 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points
model_path = "/home/exuaqiu/xuanbin/ML/tensorflow_proj/trained_model/GAN_model"


def covert_pic_to_jpeg(pic_data_path):
    """ covert all pic to jpeg """
    os.system("cd {}; mogrify -format jpeg *.*; rm -rf *.jpg".format(pic_data_path))


def resize_pic_from_data(pics_path, pic_size):
    """ standardize the pic size """
    standarlized_pics = []
    pic_batch = glob.glob(os.path.join(pics_path, "*.jpeg"))
    with tf.Session() as sess:
        for pic in pic_batch:
            image_raw_data_jpg = tf.gfile.FastGFile(pic, 'r').read()
            image_decoded = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
            resized = tf.image.resize_images(image_decoded, pic_size, method=0)
            standarlized_pics.append(tf.image.convert_image_dtype(resized, dtype=tf.float32))
    return standarlized_pics


def view_pics(pics_path):
    """ visuilize the processed pics """
    view_pic = []
    with tf.Session() as sess:
        for pic in pics_path:
            resized_pic = np.asarray(pic.eval(), dtype='uint8')
            view_pic.append(resized_pic)

    n_images = len(view_pic)
    cols = 5

    fig = plt.figure()
    for n, image in enumerate(view_pic):
        fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(image)
    plt.show()


def get_image(image_path):
    """Reads the jpg image from image_path.
    Returns the image as a tf.float32 tensor
    Args:
        image_path: tf.string tensor
    Reuturn:
        the decoded jpeg image casted to float32
    """
    content = tf.read_file(image_path)
    tf_image = tf.image.decode_jpeg(content, channels=3)
    return tf_image


def prepare_training_data(images_path, batch_size):
    """ generate training sets """
    train_input_queue = tf.train.slice_input_producer(images_path, capacity=10 * batch_size)
    batch_img = tf.train.shuffle_batch(train_input_queue, batch_size=batch_size,
                                       capacity=10 + 10 * batch_size,
                                       min_after_dequeue=10, num_threads=4)
    return batch_img


def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# Generator
def generator(x, weights, biases):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Discriminator
def discriminator(x, weights, biases):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def build_network(gen_input, disc_input, weights, biases):
    # Build Generator Network
    gen_sample = generator(gen_input, weights, biases)

    # Build 2 Discriminator Networks (one from noise input, one from generated samples)
    disc_real = discriminator(disc_input, weights, biases)
    disc_fake = discriminator(gen_sample, weights, biases)

    gen_loss = -tf.reduce_mean(tf.log(disc_fake))
    disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

    # Training Variables for each optimizer
    # By default in TensorFlow, all variables are updated by each optimizer, so we
    # need to precise for each one of them the specific variables to update.
    # Generator Network Variables
    gen_vars = [weights['gen_hidden1'], weights['gen_out'],
                biases['gen_hidden1'], biases['gen_out']]
    # Discriminator Network Variables
    disc_vars = [weights['disc_hidden1'], weights['disc_out'],
                 biases['disc_hidden1'], biases['disc_out']]

    # Build Optimizers
    optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

    return train_gen, train_disc, gen_loss, disc_loss


def train_model():
    """ start training GAN model """
    gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
    disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

    # Store layers weight & bias
    with tf.name_scope("all_weight"):
        weights = {
            'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim]), name="gen_hidden1_w"),
            'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim]), name="gen_out_w"),
            'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim]), name="disc_hidden1_w"),
            'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1]), name="disc_out_w"),
        }
    with tf.name_scope("all_biases"):
        biases = {
            'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim]), name="gen_hidden1_b"),
            'gen_out': tf.Variable(tf.zeros([image_dim]), name="gen_out_b"),
            'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim]), name="disc_hidden1_b"),
            'disc_out': tf.Variable(tf.zeros([1]), name="disc_out_b"),
        }

    train_gen, train_disc, gen_loss, disc_loss = build_network(gen_input, disc_input, weights, biases)

    standarlized_pics = resize_pic_from_data(pics_path="/home/exuaqiu/xuanbin/ML/tensorflow_proj/pic_data/special_force", pic_size=(500, 500))

    init = tf.global_variables_initializer()
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, num_steps + 1):
            # Prepare Data, Get the next batch of MNIST data (only images are needed, not labels)
            #batch_x, _ = mnist.train.next_batch(batch_size)
            batch_x = prepare_training_data(standarlized_pics, batch_size)
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
            # specify fetches, and only care about the last two, gen_loss and disc_loss
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict={disc_input: batch_x, gen_input: z})
            if i % 2000 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

        saver.save(sess, model_path)


def restore_weights_bias_for_generator(graph):

    weights = {
        'gen_hidden1': graph.get_tensor_by_name("all_weight/gen_hidden1_w:0"),
        'gen_out': graph.get_tensor_by_name("all_weight/gen_out_w:0"),
        'disc_hidden1': graph.get_tensor_by_name("all_weight/disc_hidden1_w:0"),
        'disc_out': graph.get_tensor_by_name("all_weight/disc_out_w:0"),
    }

    biases = {
        'gen_hidden1': graph.get_tensor_by_name("all_biases/gen_hidden1_b:0"),
        'gen_out': graph.get_tensor_by_name("all_biases/gen_out_b:0"),
        'disc_hidden1': graph.get_tensor_by_name("all_biases/disc_hidden1_b:0"),
        'disc_out': graph.get_tensor_by_name("all_biases/disc_out_b:0"),
    }

    return weights, biases


def test_trained_model():
    """ using the generator to generate some data for visuilize """

    gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint('./trained_model'))
        graph = tf.get_default_graph()

        weights, biases = restore_weights_bias_for_generator(graph)

        gen_sample = generator(gen_input, weights, biases)

        sess.run(init)
        # Generate images from noise, using the generator network.
        n = 4
        canvas = np.empty((28 * n, 28 * n))
        for i in range(n):
            z = 5 * np.random.uniform(-1., 1., size=[n, noise_dim])
            g = sess.run(gen_sample, feed_dict={gen_input: z})
            # Reverse colours for better display
            g = -1 * (g - 1)
            for j in range(n):
                # Draw the generated digits
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

        plt.figure(figsize=(n, n))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()


def main():
    train_model()
    #test_trained_model()
    #covert_pic_to_jpeg(pic_data_path="/home/exuaqiu/xuanbin/ML/tensorflow_proj/pic_data/special_force")
    #standarlized_pics = resize_pic_from_data(pics_path="/home/exuaqiu/xuanbin/ML/tensorflow_proj/pic_data/special_force",  pic_size=(500,500))
    #view_pics(standarlized_pics)


if __name__ == '__main__':
    main()

