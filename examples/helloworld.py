import tensorflow as tf
# simple hello world using tensorflow

hello = tf.constant("Hello, Tensorflow!")

sess = tf.Session()

print(sess.run(hello))

sess.close()