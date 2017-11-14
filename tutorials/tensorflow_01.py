# 处理结构
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

Weights = tf.Variable(tf.random_uniform([1], -1, 1.))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train_op = optimizer.minimize(loss)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

for step in range(201):
    sess.run(train_op)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))


