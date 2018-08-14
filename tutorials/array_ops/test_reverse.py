import tensorflow as tf

with tf.Session() as sess:
	a = tf.reverse([[1, 2], [3, 4]], [0])
	print(a.eval())
	a = tf.reverse([[1, 2], [3, 4]], [1])
	print(a.eval())
	a = tf.reverse([[1, 2], [3, 4]], [0, 1])
	print(a.eval())
