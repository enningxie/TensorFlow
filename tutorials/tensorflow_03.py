# Variable 变量
import tensorflow as tf

state = tf.Variable(0, name="counter")

# 定义常量one
one = tf.constant(1)

# 定义加法步骤
add_op = tf.add(state, one)

# 将state 更新成add_op
update = tf.assign(state, add_op)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

