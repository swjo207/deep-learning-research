import tensorflow as tf

with tf.Session() as sess:
	print('*mat_mul')
	a = tf.constant([[1, 2, 3], [4, 5, 6]])
	b = tf.constant([[1, 2], [1, 2], [1, 2]])
	c = tf.matmul(a, b)
	print(c.shape, c.eval())
