import tensorflow as tf

with tf.Session() as sess:
	a = tf.constant([float('nan'), float('nan'), 1])
	print(a.eval())
	b = tf.is_nan(a)
	print(b.eval())
	c = tf.is_nan([1, 2])
	print(c.eval())

