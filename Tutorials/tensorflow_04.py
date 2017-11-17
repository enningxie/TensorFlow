import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

init = tf.global_variables_initializer()  # prepare an init node

# when an InteractiveSession is created it automatically sets itself as the default session.
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()