import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 50

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # placeholder是输入数据的地方
y = tf.placeholder(tf.float32, [None, 10])  # 真实的label
# dropout的比例
keep_prob = tf.placeholder(tf.float32)

# 对数据进行重新排列，形成图像 第一维-1为输入的x数量，28,28是长和宽，1是颜色通道
x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1, 28, 28, 1

# conv1
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
# ReLU操作，输出大小为28*28*32
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
# Pooling操作，输出大小为14*14*32
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# conv2
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
# Relu操作，输出大小为14*14*64
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
# Pooling操作，输出大小为7*7*64
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# flatten
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# 全连接层1
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层2
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), 1))

optimizer = tf.train.AdamOptimizer(0.001)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x: x_batch, y: y_batch, keep_prob: 0.5})
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print('step:%d accuracy:%.4f' % (i, accuracy_value))
