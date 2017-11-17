# placeholder传入值

import tensorflow as tf
# 在tensorflow 中需要定义placeholder的type，一般为tf.float32
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 乘法操作
mul_op = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run(mul_op, feed_dict={input1: 3, input2: 4})
    print(result)

