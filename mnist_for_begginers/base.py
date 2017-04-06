import numpy as np
import tensorflow as tf

import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist)

# Noneは次元が任意の長さを取りうることを表す
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 出力
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正解ラベル
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

eta = 0.5
train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

