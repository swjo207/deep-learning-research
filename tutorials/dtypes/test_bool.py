import tensorflow as tf

with tf.Session() as sess:
	a = tf.equal(tf.eye(2), tf.eye(2))
	print(a, type(a), a.eval())
	
	a = tf.not_equal(tf.eye(2), tf.eye(2))
	print(a, type(a), a.eval())
	
	a = tf.greater(tf.eye(2), tf.ones(2))
	print(a.eval())

	a = tf.less(tf.eye(2), 1)
	print(a.eval())

	a = tf.greater_equal(tf.eye(2), 1)
	print(a.eval())

	a = tf.less_equal(tf.eye(2), 1)
	print(a.eval())
