
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

path = "/home/exuaqiu/Pictures/COD_3.png"
#img = mpimg.imread(path)
#plt.imshow(img)
#plt.show()

y = np.arange(24).reshape([2,3,4])

sess = tf.Session()
begin_y = [1, 0, 0]
size_y = [1, 2, 3]

out = tf.slice(y, begin_y, size_y)

print sess.run(out)

data = 10 * np.random.randn(1000, 4) + 1
target = np.random.randint(0, 2, 1000)

queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32], shapes=[[4], []])

# create operation
enqueue_op = queue.enqueue_many([data, target])

# create tensor
data_sample, label_sample = queue.dequeue()

# init queue runner
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

# --- using queue and coordinator to prepare some dummy data and label ---

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess=sess, coord=coord, start=True) # needs to set start to true
    for step in range(100):
        if coord.should_stop():
            break
        data_batch, label_batch = sess.run([data_sample, label_sample])
        if step % 20 == 0:
            print("Data batch is " + str(data_batch))
            print("Label batch is " + str(label_batch))

    coord.request_stop()
    coord.join(enqueue_threads)

# --- using queue and coordinator to prepare some dummy data finished---