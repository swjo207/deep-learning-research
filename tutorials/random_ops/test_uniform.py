import tensorflow as tf

with tf.Session() as sess:
	c = tf.random_uniform([2, 2])
	print(c.shape, c.eval())
