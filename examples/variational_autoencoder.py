# variational auto-encoder example

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

# Pad