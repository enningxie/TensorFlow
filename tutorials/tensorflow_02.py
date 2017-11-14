# Session 会话控制
import tensorflow as tf

# create two matrixes
matrix1 = tf.constant([[3, 3]])  # shape (1, 2)
matrix2 = tf.constant([[2], [2]])  # shape (2, 1)

product_op = tf.matmul(matrix1, matrix2)

# method 1
sess = tf.Session()
result = sess.run(product_op)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result = sess.run(product_op)
    print(result)