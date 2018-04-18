import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def maybe_download(url, filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _, _ = urllib.request.urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified ", filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "Fail to verify " + filename
        )
    return filename


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_data_set(words, vocabluary_size):

    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabluary_size-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dict


def generate_batch(data, batch_size, num_skips, skip_window):
    """
    split the data into batches for training purposes

    batch_size: size of each batch
    num_skips: amount of examples that each word will generate
    skips_window: how far can a word be associate, 1 means the word will only assocaite to the word around it

    """
    data_index = 0
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2* skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2*skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index +1) % len(data)

    for i in range(batch_size// num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            batch[i * num_skips +j] = buffer[skip_window]
            labels[i * num_skips+j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index +1) % len(data)
    return batch, labels


def generate_parametres():
    batch_size = 128
    embedding_size = 128
    skip_window = 4
    num_skips = 4
    valid_size = 16
    valid_window = 16
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64


def visulization_with_labels(low_dim_embs, labels, filename ='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels)
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha="right",va="bottom")
    plt.savefig(filename)


def main():
    url = "http://mattmahoney.net/dc/"
    vocabluary_size = 50000
    filename = maybe_download(url, 'text8.zip', 31344016)
    words = read_data(filename)
    data, count, dictionary, reversed_dict = build_data_set(words, vocabluary_size)


    batch_size = 128
    embedding_size = 128
    skip_window = 1
    num_skips = 2
    valid_size = 16
    valid_window = 16
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64


    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            """ make sure it runs on cpu, since some of the calculation is not done for GPU yet """
            embeddings = tf.Variable(tf.random_uniform([vocabluary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            nce_weights = tf.Variable(tf.truncated_normal([vocabluary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            nce_bias = tf.Variable(tf.zeros([vocabluary_size]))

        loss_func = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_bias,labels=train_labels, inputs=embed,
                                                  num_sampled=num_sampled, num_classes=vocabluary_size))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss_func)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()
        num_steps = 100001

        with tf.Session(graph=graph) as session:
            init.run()
            print("Initializing")

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
                _, loss_val = session.run([optimizer, loss_func], feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("Average loss at step ", step, " : ", average_loss)
                    average_loss = 0
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = reversed_dict[valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i,:]).argsort()[1:top_k+1]
                        log_str = "Nearest to %s : " % valid_word
                        for k in range(top_k):
                            close_word = reversed_dict[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()

    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
    plot_only = 100
    low_dims_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reversed_dict[i] for i in range(plot_only)]
    visulization_with_labels(low_dims_embs, labels)


if __name__ == '__main__':
    main()
