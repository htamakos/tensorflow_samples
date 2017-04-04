import numpy as np
import tensorflow as tf

# Model Parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# Loss
## RSS
loss = tf.reduce_sum(tf.square(linear_model - y))

# Optimizer
eta = 0.01
optimzer = tf.train.GradientDescentOptimizer(eta)
train = optimzer.minimize(loss)

# traininig_data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in rnage(1000):
    sess.run(train, { x: x_train, y: y_train })

# evaluate training accurancy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], { x: x_train, y: y_train })
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
