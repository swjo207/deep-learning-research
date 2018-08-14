import tensorflow as tf

with tf.Session() as sess:
	print('*ones')
	a = tf.ones([2, 3])
	print(a.eval())

	a = tf.ones_like(a)
	print(a.eval())

	print('*zeros')
	a = tf.zeros((2, 3))
	print(a.eval())

	print('*fill')
	a = tf.fill((2, 3), 2)
	print(a.eval())

	print('*eye')
	a = tf.eye(2)
	print(a.eval())

	a = tf.eye(2, 3)
	print(a.eval())

