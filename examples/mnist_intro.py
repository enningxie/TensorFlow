# Import MNIST

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

# Load data
X_train = mnist.train.images
Y_tarin = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

print(X_train.shape)

# get the next 64 images array and labels
batch_x, batch_y = mnist.train.next_batch(64)

print(batch_x.shape)