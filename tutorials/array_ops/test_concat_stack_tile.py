import tensorflow as tf

with tf.Session() as sess:
	print('*concat')
	a = tf.ones((2, 3))
	b = tf.zeros((2, 3))
	print(tf.concat((a, b), 0).eval())
	print(tf.concat((a, b), 1).eval())

	print('*stack')
	print(tf.stack(([1,2], [3,4])).eval())
	print(tf.stack(([1,2], [3,4]), 1).eval())
	print(tf.stack((a, b), 0).eval())
	print(tf.stack((a, b), 1).eval())

	print('*tile')
	print(tf.tile([[1,2],[3,4]], [2, 1]).eval())
	print(tf.tile([[1,2],[3,4]], [2, 3]).eval())

