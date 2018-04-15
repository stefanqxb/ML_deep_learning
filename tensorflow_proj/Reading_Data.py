
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


tf.train.shuffle_batch([img_one_queue, label_queue],
                       batch_size=batch_size,capacity =  10 + 10* batch_size,
                       min_after_dequeue = 10,
                       num_threads=16,
                       shapes=[(image_width, image_height, image_channel),()])


input_images = np.array([[0] * 784 for i in range(input_count)])
input_labels = np.array([[0] * 10 for i in range(input_count)])

# 第二次遍历图片目录是为了生成图片数据和标签
index = 0
for i in range(0, 10):
    dir = './custom_images/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            img = Image.open(filename)
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    # 通过这样的处理，使数字的线条变细，有利于提高识别准确率
                    if img.getpixel((w, h)) > 230:
                        input_images[index][w + h * width] = 0
                    else:
                        input_images[index][w + h * width] = 1
            input_labels[index][i] = 1
            index += 1