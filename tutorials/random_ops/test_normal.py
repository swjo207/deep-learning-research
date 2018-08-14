import tensorflow as tf

with tf.Session() as sess:
	c = tf.truncated_normal([2, 2], 2, 1)
	print(c.shape, c.eval())
