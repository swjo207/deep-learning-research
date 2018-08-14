import tensorflow as tf

with tf.Session() as sess:
	print('*add')
	a = tf.constant([[1, 2, 3], [4, 5, 6]])
	b = tf.constant([10, 20, 30])
	c = tf.add(a, b)
	print(c.shape, c.eval())

	print('*bias_add')
	a = tf.constant([[1, 2, 3], [4, 5, 6]])
	b = tf.constant([10, 20, 30])
	c = tf.nn.bias_add(a, b)
	print(c.shape, c.eval())

	print('*add, high rank')
	a = tf.constant([[[1,1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]])
	b = tf.constant([10, 20])
	c = tf.add(a, b)
	print(c.shape, c.eval())

	print('*bias_add, high rank')
	a = tf.constant([[[1,1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]])
	b = tf.constant([10, 20])
	c = tf.nn.bias_add(a, b)
	print(c.shape, c.eval())

