# 代码实例，添加层
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(input, input_size, output_size, activation_func=None):
    Weights = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)

    Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_func is None:
        output = Wx_plus_b
    else:
        output = activation_func(Wx_plus_b)
    return output


x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
# print(x_data.shape)
noise = np.random.normal(0, 0.005, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
print(x_data.shape)
print(noise.shape)
print(y_data.shape)
l1 = add_layer(xs, 1, 10, activation_func=tf.nn.relu)

prediction = add_layer(l1, 10, 1, activation_func=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init_op = tf.global_variables_initializer()

# 显示数据分布

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(1000):
        _, loss_ = sess.run([train_op, loss], feed_dict={xs: x_data, ys: y_data})
        if (step+1) % 50 == 0 or step == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_ = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
            lines = ax.plot(x_data, prediction_, 'r-', lw=1)
            plt.pause(0.1)
            print("Step: ", step+1, " loss: {:.4f}".format(loss_))