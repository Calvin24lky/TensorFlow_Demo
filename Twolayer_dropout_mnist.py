import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # placeholder是输入数据的地方
y = tf.placeholder(tf.float32, [None, 10])  # 真实的label
keep_prob = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
# W1 = tf.Variable(tf.zeros([784, 300])) 全零会陷入局部最优解
b1 = tf.Variable(tf.zeros([300]))
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
W2 = tf.Variable(tf.zeros([300, 10]))
b2 = tf.Variable(tf.zeros([10]))
y_pred = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)  # matmul=matrix multiply矩阵相乘


loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), 1))
optimizer = tf.train.AdagradOptimizer(0.3)
# optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(3000):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    # 设置dropout的比例为0.75，即只训练其1/4的参数
    sess.run(train, feed_dict={x: x_batch, y: y_batch, keep_prob: 0.75})
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 测试时dropout为1 即不使用dropout
        accuracy_value = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print('step:%d accuracy:%.4f' % (i, accuracy_value))
