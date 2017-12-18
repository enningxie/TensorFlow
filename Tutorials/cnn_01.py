import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
import matplotlib.pyplot as plt

# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# create 2 filters
filter_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filter_test[:, 3, :, 0] = 1  # vertical line
filter_test[3, :, :, 1] = 1  # horizontal line

# create a graph with input x plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filter_test, strides=[1, 2, 2, 1],padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

plt.imshow(output[0, :, :, 1])
plt.show()