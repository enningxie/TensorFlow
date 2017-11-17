import tensorflow as tf
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    # will evaluate w and x twice
    print(y.eval())  # 10
    print(z.eval())  # 15
    # efficient way to evaluate
    y_val, z_val = sess.run([y, z])
    print(y_val, z_val)