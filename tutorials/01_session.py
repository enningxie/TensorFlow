import tensorflow as tf

m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3], [3]])
dot_op = tf.matmul(m1, m2)

print("wrong_:", dot_op)  # wrong! no result

# method1 use session
sess = tf.Session()
result = sess.run(dot_op)
print("method_1:", result)
sess.close()

# method2 use session
with tf.Session() as sess:
    result_ = sess.run(dot_op)
    print("method_2:", result_)