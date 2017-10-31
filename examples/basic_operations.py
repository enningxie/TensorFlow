import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print("a: %i" % sess.run(a), "b: %i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(tf.multiply(a, b)))

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op.(define as input when running session)
# tf graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# More in details
# Matrix Multiplication from TensorFlow official tutorial

# create a constant op that produces a 1x2 matrix. the op is
# added as a node to the default graph.
# the value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])
# create another constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.], [2.]])
# create a matmul op
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result)