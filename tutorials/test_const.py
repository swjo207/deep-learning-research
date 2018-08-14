import tensorflow as tf

with tf.Session() as sess:
	a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
	print(a, type(a), a.dtype, a.shape)

	a = tf.constant([[1, 2], [3, 4]], dtype='float32')
	print(a, type(a), a.dtype, a.shape)
	a_inv = tf.matrix_inverse(a)
	print(a_inv, type(a_inv))
	a_inv_r = sess.run(a_inv)
	print(a_inv_r, type(a_inv_r))

	b_ini = tf.constant_initializer([[1, 2], [3, 4]], dtype='float32')
	print(b_ini, type(b_ini))
	b = tf.get_variable('b', [2, 2], initializer = b_ini)
	print(b.initializer, type(b.initializer))	
	#b.initializer.run() 
	sess.run(b.initializer)
	print(b, type(b), b.dtype, b.shape)
	#b_r = b.eval()
	b_r = sess.run(b)
	print(b_r, type(b_r))

	c = tf.ones([3, 4])
	print(c, type(c), c.eval())

	c = tf.zeros([3, 4])
	print(c, type(c), c.eval())

	c = tf.eye(3)
	print(c, type(c), c.eval())
