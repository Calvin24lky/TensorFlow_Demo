import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # placeholder是输入数据的地方
y = tf.placeholder(tf.float32, [None, 10])  # 真实的label


W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)  # matmul=matrix multiply矩阵相乘

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), 1))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(400):
    images, labels = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x: images, y: labels})
    if i % 25 == 0:
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('step:%d accuracy:%.4f' % (i, accuracy_value))
