import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 50
Time_Step = 28
Input_Size = 28
Cell_Num = 50

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # placeholder是输入数据的地方
y = tf.placeholder(tf.float32, [None, 10])  # 真实的label

W = tf.Variable(tf.truncated_normal([Cell_Num, 10], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[10]))

# RNN
x_input = tf.reshape(x, [-1, Time_Step, Input_Size])
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(Cell_Num)

# outputs是最后一层每个step的输出
# state是每一层最后一个step的输出
outputs, state = tf.nn.dynamic_rnn(rnn_cell, x_input, dtype=tf.float32)

# fc
y_pred = tf.nn.softmax(tf.matmul(state, W) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), 1))

optimizer = tf.train.AdamOptimizer(0.001)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict={x: x_batch, y: y_batch})
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('step:%d accuracy:%.4f' % (i, accuracy_value))
