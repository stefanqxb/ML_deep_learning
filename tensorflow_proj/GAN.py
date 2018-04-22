from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import os
import shutil
from PIL import Image

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 10000
batch_size = 128  #128 / 9
learning_rate = 0.00002

pic_size = 28 #128

# Network Params
image_dim = 784 # 128*128 pixels  # 784 28*28
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100  # Noise data points
model_path = "/home/exuaqiu/xuanbin/ML/tensorflow_proj/trained_model/GAN_model"


# ---  preparation of pics --- #

def covert_pic_to_jpeg(pic_data_path):
    """ covert all pic to jpeg """
    os.system("cd {}; mogrify -format jpeg *.*; rm -rf *.jpg".format(pic_data_path))


def resize_pic_from_data(pics_path, pic_size):
    """ standardize the pic size """
    tmp_path = os.path.join(pics_path, "tmp")
    tmp_pics = []
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path)

    pic_batch = glob.glob(os.path.join(pics_path, "*.jpeg"))
    with tf.Session() as sess:
        for index, pic in enumerate(pic_batch):
            image_raw_data_jpg = tf.gfile.FastGFile(pic, 'r').read()
            image_decoded = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
            resized = tf.image.resize_images(image_decoded, pic_size, method=0)
            new_image = np.asarray(resized.eval(), dtype='uint8')
            encoded_image = tf.image.encode_jpeg(new_image)
            new_image_path = os.path.join(tmp_path, str(index) + ".jpeg")
            with tf.gfile.GFile(new_image_path, "wb") as f:
                f.write(encoded_image.eval())
            tmp_pics.append(new_image_path)

    return tmp_pics


def prepare_training_data_set(pic_set, tf_size):
    """ reformat the pic with array and pixels value """
    tf_pics = np.array([[0] * tf_size for _ in range(len(pic_set))])

    for index, pic in enumerate(pic_set):
        pix_index = 0
        im = Image.open(pic)
        pix = im.load()
        width, height = im.size

        for h in range(0, height):
            for w in range(0, width):
                # convert the rgb value to one unique value to represent the color
                #tf_pics[index][pix_index] = 65536 * pix[w, h][0] + 256*pix[w, h][1] + pix[w, h][2]
                tf_pics[index][pix_index] = 0.0008 * pix[w, h][0] + 0.0005 * pix[w, h][1] + 0.0003 * pix[w, h][2]
                pix_index += 1
            pix_index += 1

        with tf.Session() as sess:
            tf_pics = tf_pics.astype(dtype="float32")
            tf.norm(tf_pics, axis=1)
    return tf_pics


def view_pics(pics_path):
    """ visuilize the processed pics """
    view_pic = []
    with tf.Session() as sess:
        for pic in pics_path:
            image_raw_data_jpg = tf.gfile.FastGFile(pic, 'r').read()
            image_decoded = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
            resized_pic = np.asarray(image_decoded.eval(), dtype='uint8')
            view_pic.append(resized_pic)

    n_images = len(view_pic)
    cols = 5

    fig = plt.figure()
    for n, image in enumerate(view_pic):
        fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(image)
    plt.show()


def radmon_select_training_data(images_set, batch_size):
    """ ramdonly pick training data set """
    batch_img = images_set[np.random.randint(images_set.shape[0], size=batch_size), :]
    return batch_img


# ---  structure GAN model --- #


def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


def add_hidden_layer(input_x, weight, bias):
    """ add hidden layer """
    hidden_layer = tf.matmul(input_x, weight)
    hidden_layer = tf.add(hidden_layer, bias)
    hidden_layer = tf.nn.relu(hidden_layer)
    return hidden_layer


def add_output_layer(input, weight, bias):
    """ add output layer """
    out_layer = tf.matmul(input, weight)
    out_layer = tf.add(out_layer, bias)
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Generator
def generator(x, weights, biases):

    hidden_layer = add_hidden_layer(x, weights['gen_hidden1'], biases['gen_hidden1'])
    out_layer = add_output_layer(hidden_layer, weights['gen_out'], biases['gen_out'])
    return out_layer


# Discriminator
def discriminator(x, weights, biases):
    hidden_layer = add_hidden_layer(x, weights['disc_hidden1'], biases['disc_hidden1'])
    out_layer = add_output_layer(hidden_layer, weights['disc_out'], biases['disc_out'])
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


def train_model(pic_path=""):
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

    #tmp_pics = resize_pic_from_data(pics_path=pic_path, pic_size=(1, image_dim))
    #tf_pics = prepare_training_data_set(tmp_pics, tf_size=image_dim)

    init = tf.global_variables_initializer()
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, num_steps + 1):
            # Prepare Data, Get the next batch of MNIST data (only images are needed, not labels)
            if pic_path:
                batch_x = radmon_select_training_data(tf_pics, batch_size)
            else:
                batch_x, _ = mnist.train.next_batch(batch_size)
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
            # specify fetches, and only care about the last two, gen_loss and disc_loss
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict={disc_input: batch_x, gen_input: z})
            if i % 100 == 0 or i == 1:
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
        canvas = np.empty((pic_size * n, pic_size * n))
        for i in range(n):
            z = 5 * np.random.uniform(-1., 1., size=[n, noise_dim])
            g = sess.run(gen_sample, feed_dict={gen_input: z})
            # Reverse colours for better display
            g = -1 * (g - 1)
            for j in range(n):
                # Draw the generated digits
                canvas[i * pic_size:(i + 1) * pic_size, j * pic_size:(j + 1) * pic_size] = g[j].reshape([pic_size, pic_size])

        plt.figure(figsize=(n, n))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()


def main():
    pic_path = "/home/exuaqiu/xuanbin/ML/tensorflow_proj/pic_data/dog_icons"
    #covert_pic_to_jpeg(pic_data_path=pic_path)
    train_model()
    #test_trained_model()
    #standarlized_pics = resize_pic_from_data(pics_path=pic_path,  pic_size=(64, 64))
    #view_pics(standarlized_pics)


if __name__ == '__main__':
    main()

