# tensorboard
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
# tensorboard
# a = tf.constant(2, name='a')
# b = tf.constant(3, name='b')
# x = tf.add(a, b, name='add')
#
# # writer = tf.summary.FileWriter('../graphs/lecture02', tf.get_default_graph())
#
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter('../graphs/lecture02', sess.graph)
#     print(sess.run(x))
#
# writer.close()