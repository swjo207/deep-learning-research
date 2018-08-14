import tensorflow as tf

with tf.Session() as sess:
	a_ini = tf.random_uniform_initializer()
	print(a_ini, type(a_ini))
	a = tf.get_variable('a', [4, 4], initializer = a_ini)
	print(type(a.initializer))
	a.initializer.run()
	print(a, type(a), a.eval())

	b = tf.random_uniform([4, 4])
	print(b.eval())